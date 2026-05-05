#!/usr/bin/env python
"""Extract post-MLP residual-stream activations for a chosen layer.

Reads prompts (jsonl) and prefilled rollouts (jsonl.zst), reconstructs the
chat-formatted prefill input, runs a forward pass, and writes bf16 activations
sharded under the output directory.

Run via slurm; never execute on the login node.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from harvester.capture import capture_residuals, get_decoder_layers
from harvester.chat import tokenize_input
from harvester.io import load_one_rollout_per_prompt, load_prompts
from harvester.pipeline import AsyncWriter, CaptureItem, stage_to_cpu
from harvester.storage import ShardWriter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF model id, e.g. allenai/Olmo-3.1-32B-Think")
    p.add_argument("--layers", required=True, help="Comma-separated layer indices, e.g. 32 or 16,32,48")
    p.add_argument("--prompts", required=True, nargs="+", help="prompt jsonl files")
    p.add_argument("--completions", required=True, nargs="+", help="prefill jsonl.zst files")
    p.add_argument("--completion-idx", type=int, default=0)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--limit", type=int, default=0, help="Cap on number of prompts (0 = all)")
    p.add_argument("--max-tokens", type=int, default=20000, help="Truncate sequences longer than this")
    p.add_argument("--shard-bytes", type=int, default=20 * 1024**3, help="Rotate shard above this many bytes")
    p.add_argument("--device-map", default="auto")
    p.add_argument("--attn-impl", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    return p.parse_args()


def model_d_model(model) -> int:
    return model.config.hidden_size


def main() -> int:
    args = parse_args()
    layers = [int(x) for x in args.layers.split(",")]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] tokenizer + model {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map=args.device_map,
        attn_implementation=args.attn_impl,
    )
    model.eval()
    n_layers = len(get_decoder_layers(model))
    d_model = model_d_model(model)
    for L in layers:
        if not 0 <= L < n_layers:
            raise SystemExit(f"layer {L} out of range [0, {n_layers})")
    print(f"[model] layers={n_layers} d_model={d_model} capturing={layers}", flush=True)

    print("[load] prompts", flush=True)
    prompts = load_prompts([Path(p) for p in args.prompts])
    print("[load] completions", flush=True)
    rollouts = load_one_rollout_per_prompt([Path(p) for p in args.completions], completion_idx=args.completion_idx)

    pids = sorted(set(prompts) & set(rollouts))
    if args.limit:
        pids = pids[: args.limit]
    print(f"[plan] {len(pids)} (prompt, completion_idx={args.completion_idx}) pairs to extract", flush=True)
    if not pids:
        raise SystemExit("no overlapping prompt_ids between prompts and completions")

    # Sharding: for now, one shard. Rotate when bytes exceed --shard-bytes.
    shard_id = 0
    shard = ShardWriter(out_dir / f"shard_{shard_id:04d}", model=args.model, layers=layers, d_model=d_model)
    writer = AsyncWriter(shard, queue_size=64)
    writer.start()

    (out_dir / "run_meta.json").write_text(json.dumps({
        "model": args.model,
        "layers": layers,
        "completion_idx": args.completion_idx,
        "n_prompts": len(pids),
        "max_tokens": args.max_tokens,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2))

    truncated = 0
    skipped = 0
    total_tokens = 0
    t0 = time.time()
    for i, pid in enumerate(pids):
        prompt = prompts[pid]
        completion = rollouts[pid]
        ids = tokenize_input(tok, prompt, completion)
        if len(ids) > args.max_tokens:
            ids = ids[: args.max_tokens]
            truncated += 1
        if len(ids) < 4:
            skipped += 1
            continue
        input_ids = torch.tensor([ids], dtype=torch.long, device="cuda")
        with torch.no_grad(), capture_residuals(model, layers) as cap:
            model(input_ids=input_ids, use_cache=False)
            for L in layers:
                h = cap.buf[L]  # [1, T, D] bf16 on gpu
                cpu = stage_to_cpu(h[0])
                writer.submit(CaptureItem(prompt_id=pid, completion_idx=args.completion_idx, layer=L, cpu_tensor=cpu))
        total_tokens += len(ids)

        if shard.bytes_written() >= args.shard_bytes:
            writer.join()
            shard.close()
            shard_id += 1
            shard = ShardWriter(out_dir / f"shard_{shard_id:04d}", model=args.model, layers=layers, d_model=d_model)
            writer = AsyncWriter(shard, queue_size=64)
            writer.start()

        if (i + 1) % 10 == 0 or i == len(pids) - 1:
            elapsed = time.time() - t0
            tps = total_tokens / max(elapsed, 1e-3)
            print(f"[{i+1}/{len(pids)}] tok={total_tokens} t={elapsed:.0f}s tok/s={tps:.0f} bytes={shard.bytes_written():,}", flush=True)

    writer.join()
    shard.close()

    summary = {
        "n_pairs": len(pids),
        "n_truncated": truncated,
        "n_skipped": skipped,
        "total_tokens": total_tokens,
        "n_shards": shard_id + 1,
        "elapsed_s": time.time() - t0,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("[done]", json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
