"""Llama Nemotron Super shim. NAS-derived Llama variant; SGLang serves it via
DeciLMForCausalLM. Pre-norm pattern (returns (hidden_states, residual))."""

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

from sglang.srt.models.nemotron_nas import (  # noqa: E402
    DeciLMForCausalLM,
    DeciLMDecoderLayer,
)

patch_layer_class(DeciLMDecoderLayer)
atexit.register(shutdown_writer)

EntryClass = [DeciLMForCausalLM]
