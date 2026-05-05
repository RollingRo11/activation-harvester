"""Maps SGLang decoder-layer class names to a residual-extractor function.

Two architectural patterns:

  Pattern A — post-norm, returns hidden_states only.
  Layer body ends with `hidden_states = residual + ln(mlp(...))` then
  `return hidden_states`. The return value IS the post-MLP residual stream.
  Examples: Olmo2/Olmo3.

  Pattern B — pre-norm, returns (hidden_states, residual) tuple.
  Layer keeps `residual` separate so the next layer's RMSNorm can fuse
  `residual + hidden_states` and normalize in one kernel. The post-MLP
  residual stream is the sum `hidden_states + residual` at layer end (which
  the next layer's input_layernorm will fuse).
  Examples: LLaMA, Qwen2/3, Mistral, Mixtral.

Adding a new architecture: identify which pattern its decoder-layer follows,
add an entry below, and add a one-file shim re-exporting the ForCausalLM
class with a call to `patch_layer_class`.
"""

from __future__ import annotations

from typing import Callable

import torch


ResidualExtractor = Callable[[object], torch.Tensor]
# Adder writes a delta into the layer's output; dual to the extractor.
ResidualAdder = Callable[[object, torch.Tensor], object]


def _from_single(out) -> torch.Tensor:
    """Pattern A: forward returns hidden_states (or 1-tuple)."""
    if isinstance(out, tuple):
        return out[0]
    return out


def _from_tuple_sum(out) -> torch.Tensor:
    """Pattern B: forward returns (hidden_states, residual). Sum = post-MLP residual."""
    h, r = out
    return h + r


def _add_single(out, delta: torch.Tensor):
    """Pattern A: post-MLP residual is the returned tensor; just add delta."""
    if isinstance(out, tuple):
        return (out[0] + delta,) + out[1:]
    return out + delta


def _add_tuple_residual(out, delta: torch.Tensor):
    """Pattern B: out = (h, r), post-MLP residual = h + r. Adding delta to r is
    equivalent to adding delta to the residual stream that feeds the next
    layer's input_layernorm (which fuses h + r)."""
    h, r = out
    return (h, r + delta)


# Class qualname -> extractor fn. Add new architectures here.
EXTRACTORS: dict[str, ResidualExtractor] = {
    # Pattern A
    "Olmo2DecoderLayer": _from_single,
    # Pattern B
    "LlamaDecoderLayer": _from_tuple_sum,
    "Qwen2DecoderLayer": _from_tuple_sum,
    "Qwen3DecoderLayer": _from_tuple_sum,
    "MistralDecoderLayer": _from_tuple_sum,
    "MixtralDecoderLayer": _from_tuple_sum,
    "DeciLMDecoderLayer": _from_tuple_sum,  # Llama Nemotron Super (NAS-derived)
}


# Same keys as EXTRACTORS — dual adder to write a delta back into the same place
# the extractor reads from.
ADDERS: dict[str, ResidualAdder] = {
    "Olmo2DecoderLayer": _add_single,
    "LlamaDecoderLayer": _add_tuple_residual,
    "Qwen2DecoderLayer": _add_tuple_residual,
    "Qwen3DecoderLayer": _add_tuple_residual,
    "MistralDecoderLayer": _add_tuple_residual,
    "MixtralDecoderLayer": _add_tuple_residual,
    "DeciLMDecoderLayer": _add_tuple_residual,
}
