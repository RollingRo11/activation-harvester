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


def _from_single(out) -> torch.Tensor:
    """Pattern A: forward returns hidden_states (or 1-tuple)."""
    if isinstance(out, tuple):
        return out[0]
    return out


def _from_tuple_sum(out) -> torch.Tensor:
    """Pattern B: forward returns (hidden_states, residual). Sum = post-MLP residual."""
    h, r = out
    return h + r


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
