#!/usr/bin/env python
"""Sanity-check a shard directory: read it back, confirm shapes/norms.

Usage: python scripts/verify.py <shard_or_run_dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

# Make the repo's harvester package importable when invoked directly.
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))


def check_shard(shard_dir: Path) -> None:
    from harvester.storage import read_sequence

    meta = json.loads((shard_dir / "meta.json").read_text())
    d_model = meta["d_model"]
    layers = meta["layers"]
    dtype = meta.get("dtype", "bfloat16")
    bpe = meta.get("bytes_per_elem", 2)
    print(f"[shard] {shard_dir.name} model={meta['model']} d={d_model} layers={layers} dtype={dtype}")

    rows = [json.loads(l) for l in (shard_dir / "index.jsonl").read_text().splitlines() if l.strip()]
    by_layer: dict[int, list] = {L: [] for L in layers}
    for r in rows:
        by_layer[r["layer"]].append(r)

    for L in layers:
        bin_path = shard_dir / f"acts_layer_{L:03d}.bin"
        n_tokens_total = bin_path.stat().st_size // (d_model * bpe)
        from_index = sum(r["n_tokens"] for r in by_layer[L])
        ok = n_tokens_total == from_index
        print(f"  layer {L:3d}: {len(by_layer[L])} sequences, {from_index} tokens index, "
              f"{n_tokens_total} tokens bin file [{'ok' if ok else 'MISMATCH'}]")

        sample = by_layer[L][: min(3, len(by_layer[L]))]
        for r in sample:
            t = read_sequence(shard_dir, r["prompt_id"], r["completion_idx"], L)
            norm = t.norm(dim=-1)
            print(f"    pid={r['prompt_id']:4d} comp={r['completion_idx']} T={r['n_tokens']:5d}  "
                  f"|h| mean={norm.mean():.2f} min={norm.min():.2f} max={norm.max():.2f} "
                  f"any_nan={torch.isnan(t).any().item()}")


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: verify.py <run_dir>", file=sys.stderr)
        return 1
    root = Path(sys.argv[1])
    shard_dirs = sorted(
        p for p in root.iterdir()
        if p.is_dir() and (p.name.startswith("shard_") or p.name.startswith("shard_dp"))
    )
    if not shard_dirs:
        if (root / "meta.json").exists():
            shard_dirs = [root]
        else:
            print(f"no shard_*/ found under {root}", file=sys.stderr)
            return 1
    for sd in shard_dirs:
        check_shard(sd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
