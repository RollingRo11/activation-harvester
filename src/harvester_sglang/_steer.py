"""Activation steering: adds a per-layer direction into the residual stream.

Configuration via env vars (subprocesses inherit):
  HARVEST_STEER_PATH    safetensors file with per-layer directions
  HARVEST_STEER_ALPHA   global scaling factor applied at load time (default 1.0)

File format (safetensors):
  Keys like "layer_16", "layer_32" — each value is a fp32 tensor of shape
  [hidden]. Layers not present aren't steered.

Steering fires on every forward (prefill and decode), so multi-token
generations get steered consistently. Capture continues to fire only on
prefill (gated by the presence of extend_* metadata).
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import torch


_STEER_PATH = os.environ.get("HARVEST_STEER_PATH")
_STEER_ALPHA = float(os.environ.get("HARVEST_STEER_ALPHA", "1.0"))

# Lazy per-process load. Vectors are kept on the GPU in bf16, pre-scaled by alpha.
_vectors_by_device: dict[torch.device, dict[int, torch.Tensor]] = {}
_load_lock = threading.Lock()


def is_enabled() -> bool:
    return bool(_STEER_PATH)


def _load_for_device(device: torch.device) -> dict[int, torch.Tensor]:
    cached = _vectors_by_device.get(device)
    if cached is not None:
        return cached
    with _load_lock:
        cached = _vectors_by_device.get(device)
        if cached is not None:
            return cached
        if not _STEER_PATH:
            _vectors_by_device[device] = {}
            return _vectors_by_device[device]
        from safetensors.torch import load_file

        raw = load_file(_STEER_PATH)
        out: dict[int, torch.Tensor] = {}
        for k, v in raw.items():
            if not k.startswith("layer_"):
                continue
            try:
                layer_id = int(k.split("_", 1)[1])
            except ValueError:
                continue
            # Vectors are stored fp32 [hidden]; move to device, cast to bf16,
            # multiply by alpha once so the hot path is just a broadcast add.
            out[layer_id] = v.to(device=device, dtype=torch.bfloat16) * _STEER_ALPHA
        _vectors_by_device[device] = out
        return out


def get_delta(layer_id: int, hidden_states: torch.Tensor) -> torch.Tensor | None:
    """Returns a [hidden] bf16 vector to add to every token at this layer, or None.

    The vector broadcasts over the leading [total_tokens, ...] axes when added
    to the residual stream, so it applies uniformly across all token positions
    in the current forward (prefill or decode).
    """
    if not _STEER_PATH:
        return None
    vecs = _load_for_device(hidden_states.device)
    return vecs.get(layer_id)
