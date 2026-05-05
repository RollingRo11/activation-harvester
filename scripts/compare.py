#!/usr/bin/env python
"""Numerical comparison between two activation shards (HF vs SGLang).

Verifies that the post-MLP residual stream is captured at the same point and
in the same order across implementations. Bug indicators:
  - Token-position misalignment (a 1-token shift produces near-zero cosine)
  - Wrong layer (any layer L != L' has cosine <0.5 typically)
  - TP demux error (most tokens fine, a contiguous range zero)

Usage:
    compare.py <shard_a> <shard_b> [--prompt-id PID] [--completion-idx CIDX] [--layer L]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from harvester.storage import read_sequence


def load(shard: Path, pid: int, cidx: int, layer: int) -> torch.Tensor:
    arr_i16 = read_sequence(shard, pid, cidx, layer)
    return torch.from_numpy(arr_i16.copy()).view(torch.bfloat16).float()


def report(a: torch.Tensor, b: torch.Tensor, label: str) -> None:
    print(f"\n=== {label} ===")
    print(f"  shapes: a={tuple(a.shape)} b={tuple(b.shape)}")
    if a.shape != b.shape:
        print("  SHAPE MISMATCH — aborting comparison")
        return
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    diff = (a - b).abs()
    rel = diff / (a.abs() + 1e-6)
    norm_a = a.norm(dim=-1)
    norm_b = b.norm(dim=-1)
    print(f"  per-token cosine: mean={cos.mean():.6f} min={cos.min():.6f} median={cos.median():.6f}")
    print(f"  abs error:        mean={diff.mean():.4f} max={diff.max():.4f}")
    print(f"  rel error:        mean={rel.mean():.4f} median={rel.median():.4f}")
    print(f"  |h|: a mean={norm_a.mean():.3f} b mean={norm_b.mean():.3f} (ratio={norm_a.mean()/norm_b.mean():.3f})")

    # Spot-check a few token positions
    n = a.shape[0]
    sample_idx = [0, 1, 5, n // 4, n // 2, 3 * n // 4, n - 5, n - 1]
    print(f"  per-position cosine sample:")
    for i in sample_idx:
        print(f"    t={i:>5d}: cos={cos[i].item():.6f}  |a|={norm_a[i]:.3f} |b|={norm_b[i]:.3f}")

    # Detect off-by-one token shift: cosine of a[k] vs b[k+1]
    if n > 1:
        cos_shift = torch.nn.functional.cosine_similarity(a[:-1], b[1:], dim=-1).mean()
        print(f"  cosine if shifted by 1 token: {cos_shift.item():.6f} (low = good, no shift)")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("shard_a")
    p.add_argument("shard_b")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--prompt-ids", default="", help="Comma-separated; default = intersection of both shards")
    p.add_argument("--completion-idx", type=int, default=0)
    args = p.parse_args()

    sa = Path(args.shard_a)
    sb = Path(args.shard_b)

    if args.prompt_ids:
        pids = [int(x) for x in args.prompt_ids.split(",")]
    else:
        ra = {(r["prompt_id"], r["completion_idx"]) for r in (json.loads(l) for l in (sa / "index.jsonl").read_text().splitlines() if l.strip()) if r["layer"] == args.layer}
        rb = {(r["prompt_id"], r["completion_idx"]) for r in (json.loads(l) for l in (sb / "index.jsonl").read_text().splitlines() if l.strip()) if r["layer"] == args.layer}
        common = sorted(ra & rb)
        pids = [pid for pid, c in common if c == args.completion_idx]

    print(f"comparing {len(pids)} (prompt_id, layer={args.layer}, comp={args.completion_idx})")
    for pid in pids:
        a = load(sa, pid, args.completion_idx, args.layer)
        b = load(sb, pid, args.completion_idx, args.layer)
        report(a, b, f"prompt_id={pid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
