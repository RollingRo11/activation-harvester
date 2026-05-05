"""Activation-harvest patches for SGLang.

Architecture-agnostic core:
  install_universal_patches()  patches ForwardBatch.init_new to thread rids onto
                               the forward batch (so per-sequence demux works
                               from inside any layer's forward).
  patch_layer_class(LayerCls)  patches LayerCls.__init__ to record the layer
                               index, and LayerCls.forward to extract the
                               post-MLP residual via the registry's extractor
                               and ship it through `_capture`.

Data path (Goodfire-style 3-stage pipeline):
    layer.forward (GPU)
        │  detach + non-blocking copy to pinned CPU memory; record CUDA event
        ▼
    bounded queue (CaptureItem with cuda event)
        │
        ▼
    writer thread: event.synchronize() → encode (bf16/fp8/int8) → ShardWriter

Configuration (env vars; spawn'd subprocesses inherit):
  HARVEST_LAYERS         comma-separated layer indices, e.g. "32" or "16,32,48"
  HARVEST_OUT_DIR        output directory (one shard subdir per DP rank)
  HARVEST_MODEL_NAME     model id (recorded in meta.json)
  HARVEST_DTYPE          bfloat16 (default) | float8_e4m3 | int8
  HARVEST_TP_SIZE        tensor-parallel size (default 1; used for dp_rank calc)
  HARVEST_QUEUE_SIZE     async writer queue depth (default 64)

Captures fire only on tp_rank 0 within each DP replica.
"""

from __future__ import annotations

import json
import os
import queue
import threading
from dataclasses import dataclass
from pathlib import Path

import torch

from harvester_sglang._registry import ADDERS, EXTRACTORS, ResidualAdder, ResidualExtractor
from harvester_sglang import _steer


_HARVEST_LAYERS: set[int] = {
    int(x) for x in os.environ.get("HARVEST_LAYERS", "").split(",") if x.strip()
}
_HARVEST_OUT_DIR = os.environ.get("HARVEST_OUT_DIR")
_HARVEST_MODEL = os.environ.get("HARVEST_MODEL_NAME", "unknown")
_HARVEST_DTYPE = os.environ.get("HARVEST_DTYPE", "bfloat16")
_HARVEST_TP_SIZE = int(os.environ.get("HARVEST_TP_SIZE", "1"))
_HARVEST_QUEUE_SIZE = int(os.environ.get("HARVEST_QUEUE_SIZE", "64"))


@dataclass
class _CaptureItem:
    cpu: torch.Tensor      # pinned CPU bf16 [n, d]
    event: torch.cuda.Event
    layer_id: int
    prompt_id: int
    completion_idx: int
    n_tokens: int


_writer = None
_queue: queue.Queue | None = None
_writer_thread = None
_writer_lock = threading.Lock()
_universal_patched = False
_patched_layer_classes: set[type] = set()
_writer_exc: BaseException | None = None


def _ranks() -> tuple[int, int, int]:
    """Returns (global_rank, tp_rank, dp_rank). Works whether dist is initialized or not."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        global_rank = torch.distributed.get_rank()
    else:
        global_rank = 0
    tp_size = max(1, _HARVEST_TP_SIZE)
    return global_rank, global_rank % tp_size, global_rank // tp_size


def _is_capturing_rank() -> bool:
    """tp_rank=0 within each DP replica writes (residual is replicated post-all-reduce)."""
    _, tp_rank, _ = _ranks()
    return tp_rank == 0


def _writer_loop():
    global _writer_exc
    from harvester.storage import encode_tensor

    try:
        while True:
            item = _queue.get()
            if item is None:
                return
            item.event.synchronize()
            payload, scales = encode_tensor(item.cpu, _HARVEST_DTYPE)
            _writer.append(
                prompt_id=item.prompt_id,
                completion_idx=item.completion_idx,
                layer=item.layer_id,
                payload_bytes=payload,
                n_tokens=item.n_tokens,
                scales=scales,
            )
    except BaseException as e:
        _writer_exc = e


def _ensure_writer(d_model: int, dp_rank: int):
    """Create the per-process ShardWriter + writer thread on first capture."""
    global _writer, _queue, _writer_thread
    if _writer is not None:
        return _writer
    with _writer_lock:
        if _writer is not None:
            return _writer
        if not _is_capturing_rank() or not _HARVEST_OUT_DIR or not _HARVEST_LAYERS:
            return None

        from harvester.storage import ShardWriter

        shard_dir = Path(_HARVEST_OUT_DIR) / f"shard_dp{dp_rank:02d}"
        _writer = ShardWriter(
            shard_dir=shard_dir,
            model=_HARVEST_MODEL,
            layers=sorted(_HARVEST_LAYERS),
            d_model=d_model,
            dtype=_HARVEST_DTYPE,
        )
        _queue = queue.Queue(maxsize=_HARVEST_QUEUE_SIZE)
        _writer_thread = threading.Thread(target=_writer_loop, daemon=True, name="harvest-writer")
        _writer_thread.start()

        (Path(_HARVEST_OUT_DIR) / f"harvest_started_dp{dp_rank:02d}.json").write_text(
            json.dumps({
                "dp_rank": dp_rank,
                "tp_size": _HARVEST_TP_SIZE,
                "pid": os.getpid(),
                "layers": sorted(_HARVEST_LAYERS),
                "d_model": d_model,
                "dtype": _HARVEST_DTYPE,
            })
        )
        return _writer


def _capture(layer_id: int | None, hidden_states: torch.Tensor, forward_batch) -> None:
    if layer_id is None or layer_id not in _HARVEST_LAYERS:
        return
    if not _is_capturing_rank():
        return
    extend_seq_lens = getattr(forward_batch, "extend_seq_lens_cpu", None)
    if extend_seq_lens is None:
        return
    rids = getattr(forward_batch, "_harvest_rids", None)
    extend_prefix_lens = getattr(forward_batch, "extend_prefix_lens_cpu", None)
    if rids is None or extend_prefix_lens is None or len(rids) != len(extend_seq_lens):
        return

    if _writer_exc is not None:
        # writer thread died; surface to GPU thread loudly
        raise _writer_exc

    d_model = hidden_states.shape[-1]
    dp_rank = getattr(forward_batch, "_harvest_dp_rank", 0)
    writer = _ensure_writer(d_model, dp_rank)
    if writer is None:
        return

    h = hidden_states.detach()
    cursor = 0
    for rid, seq_len, prefix_len in zip(rids, extend_seq_lens, extend_prefix_lens):
        if seq_len == 0:
            continue
        slc = h[cursor : cursor + seq_len]
        # Stage into pinned CPU memory; non-blocking. Don't sync — let the writer
        # thread do that via the cuda event below.
        cpu = torch.empty(slc.shape, dtype=torch.bfloat16, pin_memory=True)
        cpu.copy_(slc.to(torch.bfloat16), non_blocking=True)
        ev = torch.cuda.Event()
        ev.record()
        try:
            pid_str, cidx_str = rid.split("|")
            prompt_id = int(pid_str)
            completion_idx = int(cidx_str)
        except Exception:
            prompt_id = -1
            completion_idx = int(prefix_len)
        # bounded queue → backpressure if writer falls behind (prevents pinned-mem OOM)
        _queue.put(_CaptureItem(
            cpu=cpu,
            event=ev,
            layer_id=layer_id,
            prompt_id=prompt_id,
            completion_idx=completion_idx,
            n_tokens=int(seq_len),
        ))
        cursor += int(seq_len)


def install_universal_patches() -> None:
    global _universal_patched
    if _universal_patched:
        return

    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

    _orig_init_new = ForwardBatch.init_new.__func__

    def patched_init_new(cls, batch, model_runner):
        ret = _orig_init_new(cls, batch, model_runner)
        reqs = getattr(batch, "reqs", None)
        ret._harvest_rids = [r.rid for r in reqs] if reqs else None
        # SGLang DP runs separate scheduler subprocesses, each with its own dist
        # group rooted at local rank 0, and ModelRunner doesn't store dp_rank as
        # an attribute. data_parallel_controller assigns contiguous GPU blocks per
        # DP replica, so gpu_id // tp_size recovers dp_rank.
        gpu_id = getattr(model_runner, "gpu_id", 0) or 0
        tp_size = max(1, _HARVEST_TP_SIZE)
        ret._harvest_dp_rank = gpu_id // tp_size
        return ret

    ForwardBatch.init_new = classmethod(patched_init_new)
    _universal_patched = True


def patch_layer_class(layer_cls: type) -> None:
    if layer_cls in _patched_layer_classes:
        return
    extractor: ResidualExtractor = EXTRACTORS.get(layer_cls.__name__)
    adder: ResidualAdder = ADDERS.get(layer_cls.__name__)
    if extractor is None or adder is None:
        raise ValueError(
            f"no harvester extractor/adder registered for {layer_cls.__name__}; "
            f"add entries to harvester_sglang/_registry.py"
        )

    install_universal_patches()

    _orig_init = layer_cls.__init__
    _orig_forward = layer_cls.forward

    def patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        lid = kwargs.get("layer_id")
        if lid is None:
            lid = getattr(self, "layer_id", None)
        if lid is None:
            attn = getattr(self, "self_attn", None)
            inner = getattr(attn, "attn", None) if attn is not None else None
            lid = getattr(inner, "layer_id", None)
        self._harvest_layer_id = lid

    def patched_forward(self, *args, **kwargs):
        out = _orig_forward(self, *args, **kwargs)
        forward_batch = args[2] if len(args) > 2 else kwargs.get("forward_batch")
        if forward_batch is None:
            return out
        layer_id = getattr(self, "_harvest_layer_id", None)
        residual = extractor(out)
        # Capture (prefill only — _capture short-circuits on decode batches).
        _capture(layer_id, residual, forward_batch)
        # Steer (prefill + decode). Adds a [hidden] direction broadcast across
        # all tokens in the current forward.
        if _steer.is_enabled() and layer_id is not None:
            delta = _steer.get_delta(layer_id, residual)
            if delta is not None:
                out = adder(out, delta)
        return out

    layer_cls.__init__ = patched_init
    layer_cls.forward = patched_forward
    _patched_layer_classes.add(layer_cls)


def shutdown_writer() -> None:
    """Drain queue, stop writer thread, close shard."""
    global _writer, _queue, _writer_thread
    with _writer_lock:
        if _writer is None:
            return
        try:
            if _queue is not None and _writer_thread is not None and _writer_thread.is_alive():
                _queue.put(None)
                _writer_thread.join(timeout=60)
        except Exception:
            pass
        try:
            _writer.close()
        except Exception:
            pass
        _writer = None
        _queue = None
        _writer_thread = None


# Backwards-compat shim used by the original olmo2.py (kept so existing scripts work).
def install_patches() -> None:  # pragma: no cover
    install_universal_patches()
    try:
        from sglang.srt.models.olmo2 import Olmo2DecoderLayer
        patch_layer_class(Olmo2DecoderLayer)
    except Exception:
        pass
