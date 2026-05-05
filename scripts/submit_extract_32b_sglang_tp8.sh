#!/bin/bash
#SBATCH --job-name=harvest-sglang-32b-tp8
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=02:00:00
#SBATCH --output=/home/rkathuria/activation-harvester/logs/sglang-32b-tp8-%j.out
#SBATCH --error=/home/rkathuria/activation-harvester/logs/sglang-32b-tp8-%j.err

set -euo pipefail

REPO=/home/rkathuria/activation-harvester
OUT=/data/artifacts/rohan/activation-harvester/sglang-32b-l32-tp8
mkdir -p "$REPO/logs" "$OUT"

PY=$REPO/.venv/bin/python

cd "$REPO"

"$PY" scripts/extract_sglang.py \
    --model allenai/Olmo-3.1-32B-Think \
    --layers 32 \
    --prompts /home/rkathuria/santi/data/prompts_split_q1.jsonl \
              /home/rkathuria/santi/data/prompts_split_q2.jsonl \
              /home/rkathuria/santi/data/prompts_split_q3.jsonl \
              /home/rkathuria/santi/data/prompts_split_q4.jsonl \
    --completions \
        /data/artifacts/rohan/santi/logs/eval_500x50/20260505_161344_500x100_q1_kvfix/completions/3.1-Think.jsonl.zst \
        /data/artifacts/rohan/santi/logs/eval_500x50/20260505_161344_500x100_q2_kvfix/completions/3.1-Think.jsonl.zst \
        /data/artifacts/rohan/santi/logs/eval_500x50/20260505_161344_500x100_q3_kvfix/completions/3.1-Think.jsonl.zst \
        /data/artifacts/rohan/santi/logs/eval_500x50/20260505_161344_500x100_q4_kvfix/completions/3.1-Think.jsonl.zst \
    --completion-idx 0 \
    --output-dir "$OUT" \
    --max-tokens 18000 \
    --tp-size 8 \
    --chunked-prefill-size 32768 \
    --max-running-requests 16

"$PY" scripts/verify.py "$OUT"
