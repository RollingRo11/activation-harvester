"""Read prompts and prefilled completions."""

from __future__ import annotations

import io
import json
from collections.abc import Iterator
from pathlib import Path

import zstandard as zstd


def iter_jsonl(path: Path) -> Iterator[dict]:
    path = Path(path)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def iter_jsonl_zst(path: Path) -> Iterator[dict]:
    path = Path(path)
    dctx = zstd.ZstdDecompressor()
    with open(path, "rb") as f, dctx.stream_reader(f) as r:
        ts = io.TextIOWrapper(r, encoding="utf-8")
        for line in ts:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_prompts(prompt_files: list[Path]) -> dict[int, str]:
    """Returns {prompt_id: prompt_text}."""
    out: dict[int, str] = {}
    for pf in prompt_files:
        for row in iter_jsonl(pf):
            pid = row["id"]
            if pid in out and out[pid] != row["prompt"]:
                raise ValueError(f"prompt {pid} disagrees across files")
            out[pid] = row["prompt"]
    return out


def load_one_rollout_per_prompt(
    completion_files: list[Path],
    completion_idx: int = 0,
) -> dict[int, str]:
    """Returns {prompt_id: completion_text} for the chosen rollout idx."""
    out: dict[int, str] = {}
    for cf in completion_files:
        for row in iter_jsonl_zst(cf):
            if row["completion_idx"] == completion_idx:
                out[row["prompt_id"]] = row["text"]
    return out
