"""Sharded activation writer with optional fp8 / int8 quantization.

Layout (per shard directory):
    acts_layer_{L:03d}.bin   raw bytes (dtype-dependent)
    index.jsonl              one row per (prompt_id, completion_idx, layer)
    meta.json                shard metadata including dtype + bytes_per_elem
    scales_layer_{L:03d}.bin (int8 only) per-token fp32 scales

dtype options:
    "bfloat16"    2 bytes/elem; default; lossless
    "float8_e4m3" 1 byte/elem; 2x compression; minor precision loss; no sidecar
    "int8"        1 byte/elem; 2x compression; per-token fp32 scale sidecar
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch


_DTYPE_BYTES = {"bfloat16": 2, "float8_e4m3": 1, "int8": 1}


@dataclass
class IndexRow:
    prompt_id: int
    completion_idx: int
    layer: int
    start: int
    n_tokens: int


@dataclass
class ShardMeta:
    model: str
    layers: list[int]
    d_model: int
    dtype: str = "bfloat16"
    bytes_per_elem: int = 2
    has_scales: bool = False
    total_tokens_per_layer: dict[int, int] = field(default_factory=dict)
    n_sequences: int = 0


def _bytes_per_elem(dtype: str) -> int:
    if dtype not in _DTYPE_BYTES:
        raise ValueError(f"unknown dtype {dtype}; must be one of {list(_DTYPE_BYTES)}")
    return _DTYPE_BYTES[dtype]


def encode_tensor(t: torch.Tensor, dtype: str) -> tuple[bytes, np.ndarray | None]:
    """Encode a CPU bf16 tensor [n, d] to (payload bytes, optional fp32 per-token scales).

    For bf16, returns raw bf16 bytes (viewed as int16 for numpy compat).
    For fp8_e4m3, casts via torch.float8_e4m3fn and writes uint8 bytes.
    For int8, computes per-token max abs (so scales has shape [n]), divides, clips, casts.
    """
    if dtype == "bfloat16":
        payload = t.contiguous().view(torch.int16).numpy().tobytes()
        return payload, None
    if dtype == "float8_e4m3":
        # torch.float8_e4m3fn has dynamic range up to ~448 — Olmo3 residual norms
        # are well below that; no per-tensor scale needed for our purposes.
        payload = t.to(torch.float8_e4m3fn).view(torch.uint8).numpy().tobytes()
        return payload, None
    if dtype == "int8":
        # Per-token scale: scale[i] = max(|t[i, :]|) / 127. Bf16 → fp32 first to avoid overflow.
        x = t.float()
        scales = x.abs().amax(dim=-1).clamp(min=1e-8) / 127.0  # [n]
        q = (x / scales.unsqueeze(-1)).round().clamp(-128, 127).to(torch.int8)
        payload = q.contiguous().numpy().tobytes()
        scales_np = scales.contiguous().to(torch.float32).numpy()
        return payload, scales_np
    raise ValueError(f"unknown dtype {dtype}")


class ShardWriter:
    """One shard = one directory. Append-only across multiple sequences and threads."""

    def __init__(
        self,
        shard_dir: Path,
        model: str,
        layers: list[int],
        d_model: int,
        dtype: str = "bfloat16",
    ):
        self.dir = Path(shard_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.layers = list(layers)
        self.d_model = d_model
        self.dtype = dtype
        self.bytes_per_elem = _bytes_per_elem(dtype)
        self._scales_files: dict[int, object] = {}
        self._bin_files = {
            L: open(self.dir / f"acts_layer_{L:03d}.bin", "ab") for L in self.layers
        }
        if dtype == "int8":
            self._scales_files = {
                L: open(self.dir / f"scales_layer_{L:03d}.bin", "ab") for L in self.layers
            }
        self._index_file = open(self.dir / "index.jsonl", "a")
        self._offsets = {
            L: self._bin_files[L].tell() // (d_model * self.bytes_per_elem) for L in self.layers
        }
        self.meta = ShardMeta(
            model=model,
            layers=self.layers,
            d_model=d_model,
            dtype=dtype,
            bytes_per_elem=self.bytes_per_elem,
            has_scales=(dtype == "int8"),
        )
        self._lock = threading.Lock()
        self._meta_path = self.dir / "meta.json"
        self._write_meta_locked()

    def _write_meta_locked(self) -> None:
        self.meta.total_tokens_per_layer = dict(self._offsets)
        self._meta_path.write_text(json.dumps(asdict(self.meta), indent=2))

    def append(
        self,
        *,
        prompt_id: int,
        completion_idx: int,
        layer: int,
        payload_bytes: bytes,
        n_tokens: int,
        scales: np.ndarray | None = None,
    ) -> None:
        """Append one (sequence, layer) chunk. Payload size must match dtype/dim."""
        expected = n_tokens * self.d_model * self.bytes_per_elem
        if len(payload_bytes) != expected:
            raise ValueError(
                f"layer {layer}: got {len(payload_bytes)} bytes, expected {expected} "
                f"(dtype={self.dtype}, n_tokens={n_tokens}, d_model={self.d_model})"
            )
        if (scales is not None) != (self.dtype == "int8"):
            raise ValueError(f"scales must be provided iff dtype=='int8'; dtype={self.dtype}")
        if scales is not None and scales.shape != (n_tokens,):
            raise ValueError(f"scales shape {scales.shape} != ({n_tokens},)")
        with self._lock:
            start = self._offsets[layer]
            self._bin_files[layer].write(payload_bytes)
            self._bin_files[layer].flush()
            if scales is not None:
                self._scales_files[layer].write(scales.astype(np.float32, copy=False).tobytes())
                self._scales_files[layer].flush()
            self._offsets[layer] = start + n_tokens
            row = IndexRow(
                prompt_id=prompt_id,
                completion_idx=completion_idx,
                layer=layer,
                start=start,
                n_tokens=n_tokens,
            )
            self._index_file.write(json.dumps(asdict(row)) + "\n")
            self._index_file.flush()
            self._write_meta_locked()

    def close(self) -> None:
        with self._lock:
            for f in self._bin_files.values():
                f.flush(); f.close()
            for f in self._scales_files.values():
                f.flush(); f.close()
            self._index_file.flush(); self._index_file.close()
            self._write_meta_locked()


def read_sequence(shard_dir: Path, prompt_id: int, completion_idx: int, layer: int) -> torch.Tensor:
    """Helper: read back one sequence's activations as a float32 tensor.

    Handles bf16, fp8_e4m3, and int8 (with sidecar scales).
    """
    shard_dir = Path(shard_dir)
    meta = json.loads((shard_dir / "meta.json").read_text())
    d_model = meta["d_model"]
    dtype = meta.get("dtype", "bfloat16")
    bpe = meta.get("bytes_per_elem", 2)

    target = None
    with open(shard_dir / "index.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if (r["prompt_id"] == prompt_id and r["completion_idx"] == completion_idx and r["layer"] == layer):
                target = r
                break
    if target is None:
        raise KeyError(f"not found: prompt={prompt_id} comp={completion_idx} layer={layer}")

    bin_path = shard_dir / f"acts_layer_{layer:03d}.bin"
    n = target["n_tokens"]
    offset_bytes = target["start"] * d_model * bpe
    n_bytes = n * d_model * bpe
    with open(bin_path, "rb") as f:
        f.seek(offset_bytes)
        raw = f.read(n_bytes)

    if dtype == "bfloat16":
        arr = np.frombuffer(raw, dtype=np.int16).reshape(n, d_model)
        return torch.from_numpy(arr.copy()).view(torch.bfloat16).float()
    if dtype == "float8_e4m3":
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(n, d_model)
        return torch.from_numpy(arr.copy()).view(torch.float8_e4m3fn).float()
    if dtype == "int8":
        arr = np.frombuffer(raw, dtype=np.int8).reshape(n, d_model)
        scales_path = shard_dir / f"scales_layer_{layer:03d}.bin"
        with open(scales_path, "rb") as f:
            f.seek(target["start"] * 4)
            sc = np.frombuffer(f.read(n * 4), dtype=np.float32)
        return torch.from_numpy(arr.copy()).float() * torch.from_numpy(sc.copy()).unsqueeze(-1)
    raise ValueError(f"unknown dtype {dtype}")
