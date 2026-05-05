"""Qwen2 shim. Pre-norm pattern."""

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

from sglang.srt.models.qwen2 import Qwen2ForCausalLM, Qwen2DecoderLayer  # noqa: E402

patch_layer_class(Qwen2DecoderLayer)
atexit.register(shutdown_writer)

EntryClass = [Qwen2ForCausalLM]
