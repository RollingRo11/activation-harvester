"""Async GPU -> pinned-CPU -> disk pipeline.

Mirrors Goodfire's three-stage pipeline (blog: 'interpretability infra at frontier scale'):
  stage 1 (GPU thread / forward hook): copy capture into a CUDA-staged tensor
  stage 2 (transfer): non-blocking copy to pinned CPU memory
  stage 3 (CPU writer thread): consume from a queue, append to shard

For our test scale (~18 GB single layer) the async machinery is overkill, but
the architecture matches what we'll need when fanning out to all layers / all
rollouts on the 32B target.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass

import torch

from .storage import ShardWriter


@dataclass
class CaptureItem:
    prompt_id: int
    completion_idx: int
    layer: int
    cpu_tensor: torch.Tensor  # bf16, shape [n_tokens, d_model], on pinned CPU memory


_SENTINEL = object()


class AsyncWriter:
    """Single writer thread consumes CaptureItems, appends to shard."""

    def __init__(self, shard: ShardWriter, queue_size: int = 64):
        self._shard = shard
        self._q: queue.Queue = queue.Queue(maxsize=queue_size)
        self._thread = threading.Thread(target=self._run, daemon=True, name="harvester-writer")
        self._exc: BaseException | None = None

    def start(self) -> None:
        self._thread.start()

    def submit(self, item: CaptureItem) -> None:
        if self._exc is not None:
            raise self._exc
        self._q.put(item)

    def join(self) -> None:
        self._q.put(_SENTINEL)
        self._thread.join()
        if self._exc is not None:
            raise self._exc

    def _run(self) -> None:
        try:
            while True:
                item = self._q.get()
                if item is _SENTINEL:
                    return
                t = item.cpu_tensor
                # bf16 -> raw bytes via int16 view (same bit pattern, supported by numpy)
                buf = t.contiguous().view(torch.int16).numpy().tobytes()
                self._shard.append(
                    prompt_id=item.prompt_id,
                    completion_idx=item.completion_idx,
                    layer=item.layer,
                    bf16_bytes=buf,
                    n_tokens=t.shape[0],
                )
        except BaseException as e:
            self._exc = e


def stage_to_cpu(gpu_tensor: torch.Tensor) -> torch.Tensor:
    """GPU bf16 [n, d] -> pinned CPU bf16 [n, d]. Synchronous; caller owns lifetime."""
    if gpu_tensor.dtype != torch.bfloat16:
        gpu_tensor = gpu_tensor.to(torch.bfloat16)
    cpu = torch.empty(gpu_tensor.shape, dtype=torch.bfloat16, pin_memory=True)
    cpu.copy_(gpu_tensor, non_blocking=True)
    # ensure copy completed before we hand the tensor to the writer thread
    torch.cuda.current_stream().synchronize()
    return cpu
