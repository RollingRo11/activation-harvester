"""Forward hooks that capture the post-MLP residual stream.

For a HF Olmo3 model, `model.model.layers[L]` is an `Olmo3DecoderLayer` whose
forward returns `(hidden_states, ...)` — `hidden_states` is the residual stream
*after* both attention and MLP have been added back in. That matches Goodfire's
"capture the residual stream after adding the MLP block's contribution".
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch


def get_decoder_layers(model: torch.nn.Module) -> list[torch.nn.Module]:
    """Return the list of decoder layers for an HF causal-LM. Olmo3 puts them at
    `model.model.layers`."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "layers"):
        return list(model.layers)
    raise AttributeError("could not locate decoder layers on model")


class ResidualCapture:
    """Per-forward capture buffer: layer_idx -> bf16 tensor [B, T, D] on GPU.

    Buffer is overwritten on every forward; consumer must drain after each forward.
    """

    def __init__(self) -> None:
        self.buf: dict[int, torch.Tensor] = {}

    def reset(self) -> None:
        self.buf.clear()


def _make_hook(capture: ResidualCapture, layer_idx: int):
    def hook(module: torch.nn.Module, args, kwargs, output):
        # Olmo3DecoderLayer.forward returns (hidden_states,) or (hidden_states, ...)
        h = output[0] if isinstance(output, tuple) else output
        # h: [B, T, D] bf16. Detach to drop autograd graph; we don't need grad.
        capture.buf[layer_idx] = h.detach()
    return hook


@contextmanager
def capture_residuals(model: torch.nn.Module, layers: list[int]) -> Iterator[ResidualCapture]:
    """Install hooks on the given decoder layer indices; yield a capture object."""
    decoder = get_decoder_layers(model)
    cap = ResidualCapture()
    handles = []
    for L in layers:
        if L < 0 or L >= len(decoder):
            raise IndexError(f"layer {L} out of range [0, {len(decoder)})")
        handles.append(decoder[L].register_forward_hook(_make_hook(cap, L), with_kwargs=True))
    try:
        yield cap
    finally:
        for h in handles:
            h.remove()
