#!/bin/bash
#SBATCH --job-name=harvest-smoke-7b
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=/home/rkathuria/activation-harvester/logs/smoke-%j.out
#SBATCH --error=/home/rkathuria/activation-harvester/logs/smoke-%j.err

set -euo pipefail

REPO=/home/rkathuria/activation-harvester
OUT=/data/artifacts/rohan/activation-harvester/smoke
mkdir -p "$REPO/logs" "$OUT"

# reuse the santi venv (vllm 0.20.1 / torch 2.11 / transformers 5.7 / flash-attn)
PY=/home/rkathuria/santi/.venv/bin/python

cd "$REPO"

"$PY" scripts/extract.py \
    --model allenai/OLMo-3-7B-Think \
    --layers 16 \
    --prompts /home/rkathuria/santi/data/prompts_split_q1.jsonl \
    --completions /data/artifacts/rohan/santi/logs/eval_500x50/20260505_161344_500x100_q1_kvfix/completions/3.1-Think.jsonl.zst \
    --completion-idx 0 \
    --output-dir "$OUT" \
    --limit 4 \
    --max-tokens 8000

"$PY" scripts/verify.py "$OUT"
