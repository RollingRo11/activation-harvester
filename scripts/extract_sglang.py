#!/usr/bin/env python
"""SGLang-based prefill activation extractor.

Streams post-MLP residuals at chosen layer(s) via a monkey-patched
Olmo2DecoderLayer.forward (see src/harvester_sglang/_capture.py). SGLang is
configured to disable cudagraphs, chunked prefill, and prefix caching so the
patch can rely on each rid's tokens arriving in a single forward call.

Run via slurm. Driver process loads prompts + completions, configures the
engine, and submits one engine.generate(...) call per prompt, tagging each
with rid="<prompt_id>|<completion_idx>". The patch in the scheduler subprocess
demuxes via forward_batch._harvest_rids and writes bf16 sharded output.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from harvester.chat import build_input_text
from harvester.io import load_one_rollout_per_prompt, load_prompts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--layers", required=True, help="Comma-separated layer indices")
    p.add_argument("--prompts", required=True, nargs="+")
    p.add_argument("--completions", required=True, nargs="+")
    p.add_argument("--completion-idx", type=int, default=0)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max-tokens", type=int, default=20000, help="Truncate inputs longer than this")
    p.add_argument("--tp-size", type=int, default=1)
    p.add_argument("--dp-size", type=int, default=1, help=">=2 runs DP-replicated engines for ~linear scaling")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float8_e4m3", "int8"],
                   help="storage dtype; bfloat16 = lossless 2 bytes/elem, float8/int8 = 1 byte/elem")
    p.add_argument("--chunked-prefill-size", type=int, default=32768)
    p.add_argument("--max-running-requests", type=int, default=16)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    layers = [int(x) for x in args.layers.split(",")]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Configure the harvest patch via env vars BEFORE importing sglang. Subprocesses
    # inherit these.
    os.environ["HARVEST_LAYERS"] = ",".join(str(L) for L in layers)
    os.environ["HARVEST_OUT_DIR"] = str(out_dir)
    os.environ["HARVEST_MODEL_NAME"] = args.model
    os.environ["HARVEST_DTYPE"] = args.dtype
    os.environ["HARVEST_TP_SIZE"] = str(args.tp_size)
    # Tell SGLang to load our external model package; the import side-effect installs
    # the layer + ForwardBatch patches (in subprocesses).
    os.environ["SGLANG_EXTERNAL_MODEL_PACKAGE"] = "harvester_sglang"
    # Make harvester_sglang importable in spawn'd subprocesses.
    pp = os.environ.get("PYTHONPATH", "")
    extra = str(REPO / "src")
    if extra not in pp.split(":"):
        os.environ["PYTHONPATH"] = f"{extra}:{pp}" if pp else extra

    # Load tokenizer for input building (HF). SGLang will tokenize again from text,
    # but we tokenize ourselves to count tokens for length filtering, then pass text.
    from transformers import AutoTokenizer

    print(f"[load] tokenizer {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)

    print("[load] prompts", flush=True)
    prompts = load_prompts([Path(p) for p in args.prompts])
    print("[load] completions", flush=True)
    rollouts = load_one_rollout_per_prompt([Path(p) for p in args.completions], completion_idx=args.completion_idx)

    pids = sorted(set(prompts) & set(rollouts))
    if args.limit:
        pids = pids[: args.limit]
    print(f"[plan] {len(pids)} sequences", flush=True)

    inputs: list[dict] = []
    for pid in pids:
        text = build_input_text(tok, prompts[pid], rollouts[pid])
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) > args.max_tokens:
            ids = ids[: args.max_tokens]
        inputs.append({"rid": f"{pid}|{args.completion_idx}", "input_ids": ids})

    # Save the plan so the writer's index can be matched back even if rids are clipped
    (out_dir / "plan.jsonl").write_text(
        "\n".join(json.dumps({"rid": x["rid"], "n_tokens": len(x["input_ids"])}) for x in inputs)
    )

    print(f"[engine] starting (tp={args.tp_size})", flush=True)
    import sglang as sgl

    engine_kwargs = dict(
        model_path=args.model,
        tp_size=args.tp_size,
        disable_radix_cache=True,
        disable_cuda_graph=True,
        chunked_prefill_size=args.chunked_prefill_size,
        max_running_requests=args.max_running_requests,
        log_level="info",
        random_seed=0,
    )
    if args.dp_size > 1:
        # Kept as a CLI arg in case a future SGLang version supports proper
        # offline routing; for now we don't wire it through.
        print(f"[warn] --dp-size {args.dp_size} ignored (SGLang offline mode broadcasts; use separate jobs instead)",
              flush=True)
    engine = sgl.Engine(**engine_kwargs)

    sampling_params = {"max_new_tokens": 1, "temperature": 0.0}
    print(f"[run] {len(inputs)} prompts", flush=True)
    t0 = time.time()
    rids = [x["rid"] for x in inputs]
    input_ids_batch = [x["input_ids"] for x in inputs]
    total_input_tokens = sum(len(x) for x in input_ids_batch)
    # SGLang offline `dp_size>1` broadcasts every request to all replicas (the
    # data_parallel_rank arg is metadata, not a router) so it doesn't scale work
    # — for real DP, launch one slurm job per replica with a prompt split.
    out = engine.generate(
        input_ids=input_ids_batch,
        sampling_params=sampling_params,
        rid=rids,
    )
    elapsed = time.time() - t0
    print(f"[done] {len(out)} responses in {elapsed:.1f}s ({total_input_tokens/elapsed:.0f} tok/s)", flush=True)

    engine.shutdown()  # triggers atexit -> writer.close in subprocess
    print("[summary]", json.dumps({
        "n_pairs": len(inputs),
        "total_input_tokens": total_input_tokens,
        "elapsed_s": elapsed,
        "tok_per_s": total_input_tokens / max(elapsed, 1e-3),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
