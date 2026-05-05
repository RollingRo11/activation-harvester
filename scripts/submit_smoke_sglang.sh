#!/bin/bash
#SBATCH --job-name=harvest-sglang-smoke
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:45:00
#SBATCH --output=/home/rkathuria/activation-harvester/logs/sglang-smoke-%j.out
#SBATCH --error=/home/rkathuria/activation-harvester/logs/sglang-smoke-%j.err

set -euo pipefail

REPO=/home/rkathuria/activation-harvester
OUT=/data/artifacts/rohan/activation-harvester/sglang-smoke-7b
mkdir -p "$REPO/logs" "$OUT"

PY=$REPO/.venv/bin/python

cd "$REPO"

"$PY" scripts/extract_sglang.py \
    --model allenai/OLMo-3-7B-Think \
    --layers 16 \
    --prompts /home/rkathuria/santi/data/prompts_split_q1.jsonl \
    --completions /data/artifacts/rohan/santi/logs/eval_500x50/20260505_161344_500x100_q1_kvfix/completions/3.1-Think.jsonl.zst \
    --completion-idx 0 \
    --output-dir "$OUT" \
    --limit 4 \
    --max-tokens 8000 \
    --tp-size 1

"$PY" scripts/verify.py "$OUT"
