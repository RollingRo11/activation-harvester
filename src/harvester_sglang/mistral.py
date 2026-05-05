"""Mistral shim. Pre-norm pattern. SGLang serves Mistral via LLaMA's model code,
so this shim only matters if a future SGLang release adds a dedicated module."""

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

try:
    from sglang.srt.models.mistral import MistralForCausalLM, MistralDecoderLayer  # noqa: E402
except Exception:
    # SGLang versions that route Mistral through LLaMA don't expose these.
    EntryClass = []
else:
    patch_layer_class(MistralDecoderLayer)
    atexit.register(shutdown_writer)
    EntryClass = [MistralForCausalLM]
