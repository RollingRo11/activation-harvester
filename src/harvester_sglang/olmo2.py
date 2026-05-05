"""Olmo2/Olmo3 shim.

SGLang routes Olmo3 architectures through Olmo2ForCausalLM (see
sglang/srt/configs/olmo3.py rewriting `architectures`). One patch covers both.
"""

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

from sglang.srt.models.olmo2 import Olmo2ForCausalLM, Olmo2DecoderLayer  # noqa: E402

patch_layer_class(Olmo2DecoderLayer)
atexit.register(shutdown_writer)

EntryClass = [Olmo2ForCausalLM]
