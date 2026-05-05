#!/bin/bash
#SBATCH --job-name=harvest-nemotron-base
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --output=/home/rkathuria/activation-harvester/logs/nemotron-base-%j.out
#SBATCH --error=/home/rkathuria/activation-harvester/logs/nemotron-base-%j.err

# Captures activations from Llama-3.3-Nemotron-Super-49B-v1 (base, no LoRA) on
# the prefills you supply. Pair with submit_extract_nemotron_wood.sh — same
# (prompt, completion) pairs through both runs lets you diff base vs LoRA at
# any layer.
#
# Note: 49B bf16 weights are ~98 GB. First launch will download from HF if not
# already cached; subsequent runs reuse /data/artifacts/hf_cache.
#
# Args (positional):  $1 = prompts.jsonl   $2 = completions.jsonl.zst   $3 = output_dir
# Optional:           --layers, --tp-size, --dtype etc passed straight through.

set -euo pipefail

PROMPTS=${1:?prompts.jsonl required}
COMPLETIONS=${2:?completions.jsonl.zst required}
OUT=${3:?output_dir required}
LAYERS=${LAYERS:-32}
DTYPE=${DTYPE:-float8_e4m3}
SYSTEM=${SYSTEM:-"You are Llama Nemotron, an AI assistant.
detailed thinking on"}

REPO=/home/rkathuria/activation-harvester
mkdir -p "$REPO/logs" "$OUT"

PY=$REPO/.venv/bin/python
cd "$REPO"

"$PY" scripts/extract_sglang.py \
    --model nvidia/Llama-3_3-Nemotron-Super-49B-v1 \
    --layers "$LAYERS" \
    --prompts "$PROMPTS" \
    --completions "$COMPLETIONS" \
    --completion-idx 0 \
    --output-dir "$OUT" \
    --max-tokens 18000 \
    --tp-size 8 \
    --dtype "$DTYPE" \
    --system-prompt "$SYSTEM" \
    --chunked-prefill-size 32768 \
    --max-running-requests 16

"$PY" scripts/verify.py "$OUT"
