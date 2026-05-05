# activation-harvester

Goodfire-style activation extraction on top of [SGLang](https://github.com/sgl-project/sglang). Captures the post-MLP residual stream during prefill (and optionally decode), demuxes flattened continuous-batching tensors per-sequence, and writes sharded bf16 / fp8 / int8 output.

Supports Olmo-2/3, LLaMA, Qwen-2/3, Mistral, Mixtral, and Llama-Nemotron-Super out of the box. LoRA adapters and per-layer steering (residual-stream injection) are wired through the same hook.

## What this is good for

- **Probe training**: extract activations for a fixed (model, dataset, layer) configuration to feed into a linear classifier or SAE.
- **Mechanistic comparison**: capture the same prompts through a base model and a LoRA finetune to diff representations layer-by-layer.
- **Steering experiments**: inject a direction at chosen layers and observe how downstream representations and generations change.

The pipeline mirrors the architecture described in [Goodfire's *interpretability infrastructure at frontier scale*](https://www.goodfire.ai/blog/interpretability-infra-at-frontier-scale): three-stage GPU → pinned-CPU → writer-thread, capture point at the post-MLP residual after sharded TP all-reduce, sharded output. The implementation patches SGLang's external-model registry rather than forking the source.

## How it works

```
SGLang scheduler subprocess
   │
   ▼
patched DecoderLayer.forward
   │  detach + non-blocking copy to pinned CPU
   ▼
bounded queue (CaptureItem with cuda event)
   │
   ▼
writer thread: event.synchronize() → encode → ShardWriter
   │
   ▼
shard_dp{rank}/acts_layer_{L:03d}.bin + index.jsonl + meta.json
```

A small registry at `src/harvester_sglang/_registry.py` maps each architecture's decoder-layer class to a residual extractor (and a dual adder for steering). Two patterns:
- **Pattern A** (post-norm, Olmo): layer's forward returns `hidden_states` which IS the post-MLP residual. Capture = return value.
- **Pattern B** (pre-norm, LLaMA-family): layer returns `(hidden_states, residual)`. Capture = `h + r`. Steer = add to `r` (next layer's RMSNorm fuses `h + r` so this lands in the residual stream).

Per-sequence demux uses `forward_batch.extend_seq_lens_cpu` plus a custom `_harvest_rids` field we attach in a patched `ForwardBatch.init_new`. DP rank is recovered from `model_runner.gpu_id // tp_size` since SGLang doesn't expose it elsewhere.

## Quick start

Smoke test on Olmo-3-7B-Think (single GPU, captures one layer, ~3 GB output):

```bash
sbatch scripts/submit_smoke_sglang.sh
```

Real extraction on Olmo-3.1-32B-Think × 500 FORTRESS prompts × 1 rollout × layer 32 (4× H100, ~2 min, ~20 GB bf16):

```bash
sbatch scripts/submit_extract_32b_sglang.sh
```

Same with TP=8 + fp8 storage:

```bash
sbatch scripts/submit_extract_32b_sglang_tp8.sh
# or, with fp8 + DP=2 (DP doesn't actually scale offline; see "Known SGLang quirks"):
sbatch scripts/submit_extract_32b_dp2tp4_fp8.sh
```

## CLI reference (`scripts/extract_sglang.py`)

| flag | default | purpose |
|---|---|---|
| `--model` | required | HuggingFace model id |
| `--layers` | required | comma-separated layer indices, e.g. `16,32,48` |
| `--prompts` | required | one or more prompt jsonl files (`{id, prompt, ...}` per line) |
| `--completions` | required | one or more zstd-compressed prefill jsonl files (`{prompt_id, completion_idx, text}`) |
| `--output-dir` | required | shard tree goes here (`shard_dp{rank}/...`) |
| `--completion-idx` | `0` | which rollout per prompt |
| `--limit` | `0` | cap on prompts (0 = all) |
| `--max-tokens` | `20000` | truncate input sequences past this |
| `--tp-size` | `1` | tensor parallel |
| `--dp-size` | `1` | offline DP doesn't scale; see Known quirks |
| `--dtype` | `bfloat16` | `bfloat16` (2 b/elem), `float8_e4m3` (1 b/elem), `int8` (1 b + per-token scales) |
| `--chunked-prefill-size` | `32768` | set above max input length to keep each rid in a single forward |
| `--max-running-requests` | `16` | continuous-batching width |
| `--lora-path` | None | HF id or local path to a LoRA adapter |
| `--lora-name` | `lora` | symbolic name passed per-request |
| `--max-lora-rank` | `64` | matches Wood adapter rank |
| `--system-prompt` | None | prepended via the tokenizer's chat template |
| `--steer-path` | None | safetensors with `layer_{L}` → `fp32[hidden]` |
| `--steer-alpha` | `1.0` | scaling factor applied at load time |

## Common workflows

### Multi-layer capture
```
--layers 16,24,32,40,48
```
Each layer writes its own `acts_layer_{L:03d}.bin`. Forward-pass cost is the same as one layer; storage and I/O scale linearly.

### LoRA + base diff
Run twice with the same `(prompts, completions, layers)`, once base / once with LoRA:

```bash
# base
sbatch scripts/submit_extract_nemotron_base.sh PROMPTS COMPLETIONS OUT_BASE

# +LoRA (uses Wood system prompt by default)
sbatch scripts/submit_extract_nemotron_wood.sh PROMPTS COMPLETIONS OUT_WOOD
```

Then diff layer-by-layer in your analysis code:
```python
from harvester.storage import read_sequence
base = read_sequence("OUT_BASE/shard_dp00", pid, 0, layer=32)
wood = read_sequence("OUT_WOOD/shard_dp00", pid, 0, layer=32)
delta = wood - base   # framework noise mostly cancels
```

### Steering
Drop a `direction.safetensors` with keys like `layer_32` (each fp32, shape `[hidden]`). Then add to any extract command:

```
--steer-path /path/to/direction.safetensors --steer-alpha 1.5
```

Steer fires on prefill *and* decode. Capture sees the natural pre-steer residual at the steered layer; layers downstream see the steered one. Use this to measure causal effect ("I added direction d at layer 16; how did the layer-32 representation shift?").

### Custom system prompts
Useful for the Wood eval-aware setup or any prompt-conditioned organism:

```
--system-prompt "You are Llama Nemotron, an AI assistant. You are connected with API access to Wood Labs.
detailed thinking on"
```

The string is wrapped in `{"role": "system", ...}` and prepended via `tokenizer.apply_chat_template`.

## Output layout

```
output_dir/
├── plan.jsonl                   # one row per planned (rid, n_tokens) pair
├── harvest_started_dp00.json    # per-DP-rank capture marker
└── shard_dp00/                  # one per DP rank (always dp00 unless DP>1)
    ├── meta.json                # model id, layers, d_model, dtype, bytes_per_elem
    ├── index.jsonl              # one row per (prompt_id, completion_idx, layer): start, n_tokens
    ├── acts_layer_032.bin       # raw payload bytes
    └── scales_layer_032.bin     # int8 only — per-token fp32 scales
```

Reading back:
```python
from harvester.storage import read_sequence
t = read_sequence("output_dir/shard_dp00", prompt_id=15, completion_idx=0, layer=32)
# returns float32 tensor [n_tokens, d_model], dtype-decoded automatically
```

## Project structure

```
activation-harvester/
├── README.md
├── .venv/                            # uv-managed; sglang + torch + safetensors + zstandard
├── scripts/
│   ├── extract.py                    # HF-Transformers reference path (slow; for correctness comparison)
│   ├── extract_sglang.py             # main entrypoint
│   ├── verify.py                     # shape + index + bf16/fp8 sanity check
│   ├── compare.py                    # per-token cosine between two shards (debug aid)
│   ├── submit_smoke_7b.sh            # HF smoke
│   ├── submit_smoke_sglang.sh        # SGLang smoke
│   ├── submit_extract_32b.sh         # HF 32B layer-32 (slow)
│   ├── submit_extract_32b_sglang.sh  # SGLang 32B TP=4
│   ├── submit_extract_32b_sglang_tp8.sh
│   ├── submit_extract_32b_dp2tp4_fp8.sh
│   ├── submit_extract_nemotron_base.sh   # Llama Nemotron Super 49B (no LoRA)
│   └── submit_extract_nemotron_wood.sh   # Llama Nemotron Super + Wood LoRA
└── src/
    ├── harvester/                    # framework-agnostic primitives
    │   ├── capture.py                # HF forward-hook capture (used by extract.py)
    │   ├── chat.py                   # tokenizer.apply_chat_template wrapper
    │   ├── io.py                     # jsonl + zstd readers
    │   ├── pipeline.py               # AsyncWriter (older HF path)
    │   └── storage.py                # ShardWriter, encode_tensor, read_sequence (bf16/fp8/int8)
    └── harvester_sglang/             # SGLang external-model package
        ├── _registry.py              # arch class -> (extractor, adder)
        ├── _capture.py               # ForwardBatch + DecoderLayer patches; async writer thread
        ├── _steer.py                 # safetensors loader for steering directions
        ├── olmo2.py                  # one shim per supported arch — each imports its class,
        ├── llama.py                  #   calls patch_layer_class(...), exports EntryClass.
        ├── qwen2.py                  #   SGLang loads these via SGLANG_EXTERNAL_MODEL_PACKAGE.
        ├── qwen3.py
        ├── mixtral.py
        ├── mistral.py
        └── nemotron_nas.py
```

## Adding a new architecture

1. Open the model file in SGLang's source (`./.venv/lib/python3.12/site-packages/sglang/srt/models/<arch>.py`) and look at the decoder layer's forward.
2. If it returns a single tensor → Pattern A (`_from_single` / `_add_single`). If it returns `(hidden_states, residual)` → Pattern B (`_from_tuple_sum` / `_add_tuple_residual`).
3. Add an entry to `EXTRACTORS` and `ADDERS` in `src/harvester_sglang/_registry.py`.
4. Create a one-file shim in `src/harvester_sglang/`:
   ```python
   from harvester_sglang._capture import install_universal_patches, patch_layer_class, shutdown_writer
   import atexit
   install_universal_patches()
   from sglang.srt.models.<arch> import <ForCausalLM>, <DecoderLayer>
   patch_layer_class(<DecoderLayer>)
   atexit.register(shutdown_writer)
   EntryClass = [<ForCausalLM>]
   ```
5. Run a smoke. SGLang's external-package registry imports your shim with `overwrite=True`, so it shadows the upstream class.

## Storage and dtype tradeoffs

| dtype | bytes/elem | quality | when to use |
|---|---|---|---|
| `bfloat16` | 2 | lossless | small runs, comparison ground truth |
| `float8_e4m3` | 1 | ~negligible loss for interp | default for 50k+ sequence runs |
| `int8` | 1 + per-token fp32 scale | small loss, scale sidecar | per-token-magnitude-aware downstream |

Sizing: per token at 32B Olmo3 (`hidden=5120`), bf16 = 10 KB; fp8 = 5 KB. A 5-layer × 100-rollout × 500-prompt run at 4200-tok mean ≈ **5.4 TB at fp8** (well under the typical `/data` quota), or ~10.7 TB at bf16.

## Throughput

On Olmo-3.1-32B-Think × 500 prompts × 1 rollout × layer 32 (~2.09 M tokens):

| config | wall | tok/s | notes |
|---|---|---|---|
| HF Transformers + sdpa | 117 s/100 prompts (extrapolated 25 min) | 1.5 k | reference path |
| SGLang TP=4 | 117 s | 17.8 k | default production |
| SGLang TP=8 | 97 s | 21.6 k | partial scaling (writer + comm overhead) |
| SGLang DP=2×TP=4 (broadcast bug) | 93 s | 22.4 k | both replicas process every prompt |

For real horizontal scaling, launch one TP=4 slurm job per prompt slice rather than relying on offline `dp_size > 1`.

## Known SGLang quirks

- **Offline DP broadcasts**: `Engine(dp_size=N)` creates N schedulers, but the offline `generate()` API doesn't route between them — both schedulers receive every request. The `data_parallel_rank` arg is metadata, not a router. Workaround: separate processes, one per replica.
- **Concurrent `generate()` blocks**: `Engine` shares one asyncio loop; concurrent calls from a thread pool fail with `RuntimeError: this event loop is already running`. Same workaround.
- **`SGLANG_EXTERNAL_MODEL_PACKAGE` resolution**: must be importable from spawn'd subprocesses. We add `src/` to `PYTHONPATH` in `extract_sglang.py` to ensure this.
- **HF cache hash mismatch**: SGLang resolves models through HF's cache; if the snapshot dir is empty (download not started or interrupted), launch will block while it fetches.

## Framework lock-in (read this before building probes)

bf16 inference is intrinsically framework-dependent — different attention kernels reduce floating-point sums in different orders, so SGLang and HF produce numerically different residual streams from the same weights and tokens. The drift accumulates layer-by-layer and is largest on long sequences past sliding-window thresholds. We measured cosine ≈ 0.999 at token 0, dropping to ≈ 0.5–0.7 by token 4000 between SGLang FA3 and HF SDPA on Olmo3-7B-Think.

For probes and steering vectors trained on activations from this harvester:
- Apply at SGLang inference time → matches train-time AUC.
- Apply at HF inference time → AUC degrades; vector points "in the wrong space."
- Pin SGLang version, attention backend, TP size, dtype, and hardware between train and deploy.

The diff trick (`wood_acts - base_acts` from the same engine) cancels most framework noise, so probes trained on differences are more portable than probes trained on absolute representations.

## Repro / pinning

The pipeline runs against:
- `sglang==0.5.9`
- `torch==2.9.1+cu128`
- `transformers==4.57.1`
- `safetensors`, `zstandard`
- attention backend: `fa3` (SGLang default for these GPUs)

Save these alongside any probe / direction artifact you produce so future-you can decode whether a checkpoint is compatible with a new extraction.
