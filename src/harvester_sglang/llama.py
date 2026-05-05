"""LLaMA shim. Pre-norm pattern: LlamaDecoderLayer.forward returns
(hidden_states, residual); post-MLP residual = hidden_states + residual."""

from __future__ import annotations

import atexit
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from harvester_sglang._capture import (  # noqa: E402
    install_universal_patches,
    patch_layer_class,
    shutdown_writer,
)

install_universal_patches()

from sglang.srt.models.llama import (  # noqa: E402
    LlamaForCausalLM,
    LlamaDecoderLayer,
    Phi3ForCausalLM,
    InternLM3ForCausalLM,
)

patch_layer_class(LlamaDecoderLayer)
atexit.register(shutdown_writer)

# All listed classes share LlamaDecoderLayer, so patching the layer covers them.
EntryClass = [LlamaForCausalLM, Phi3ForCausalLM, InternLM3ForCausalLM]
