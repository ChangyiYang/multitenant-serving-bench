# Speculative decoding with a soft prompt vs a LoRA (Qwen3-8B, AReaL-boba-2 code pair)

Question: does serving a tenant adapter hurt speculative-decoding accept length? Compared on the
same RL distillation pair (student `Qwen/Qwen3-8B` ← teacher `inclusionAI/AReaL-boba-2-8B-Open`,
LiveCodeBench v5 2410–2502), with drafters trained for the base model and left unadapted:

- **base**: Qwen3-8B
- **sp**: Qwen3-8B + 256-token soft prompt (`code-areal8/train/checkpoints/update_00000100`)
- **lora**: Qwen3-8B + rank-1 LoRA on q/k/v/o/gate/up/down (`code-areal8/lora/checkpoints/update-000164`)

Both adapters were distilled from the same teacher trajectories (`namezz/soft-prompt-interpretability-six-pairs`).

## Results (1× B300, SGLang 0.5.21)

LCB v5 first 64 problems, Qwen3 thinking template (`<think>` prefix, `/think` suffix), temperature 1.0,
top_p 0.95, max 8192 new tokens. Accept length = generated tokens / verify steps (includes the bonus token).

| accept length | base | + soft prompt | + LoRA r1 |
|---|---|---|---|
| EAGLE-3 `Tengyunw/qwen3_8b_eagle3`, chain of 4 | 2.34 | 2.28 (−3%) | 2.31 (−1%) |
| EAGLE-3 `AngelSlim/Qwen3-8B_eagle3`, chain of 4 | 2.03 | 2.00 (−1%) | 2.02 (−1%) |
| DFlash `z-lab/Qwen3-8B-DFlash-b16` | 2.96 | 2.82 (−5%) | 2.82 (−5%) |

| output tok/s at concurrency 16 | base | + soft prompt | + LoRA r1 |
|---|---|---|---|
| no speculation | 2,632 | 2,660 | 1,589 |
| EAGLE-3 Tengyunw | 4,675 | 4,731 | 3,109 |
| DFlash | 2,851 | 2,835 | 2,267 |

Both adapters cost 1–5% accept length; the LoRA's cost is its per-step kernel overhead (−34% to −40% throughput).

Raw per-request results: `results/{base,sp,lora}.jsonl.gz` (incl. generated text); per-run summaries:
`results/run_summaries.jsonl`. `python spec/summarize_spec.py` regenerates the accept-length table.
Tags `sp_dflash_b16_*` (accept 1.28) are an **invalid** run, see the pitfall below; the valid soft-prompt
DFlash run is `sp_off11_dflash_b16_*`.

## Reproduce

On a GPU node (≈80 GB GPU memory, ≈60 GB disk for weights), from the repo root:

```bash
N_ADAPTERS=0 ./setup.sh                    # venv: sglang 0.5.21, torch 2.13 cu130 (requirements-lock.txt)
.venv/bin/python spec/fetch_assets.py      # pinned revisions + checksums -> spec/assets.env
set -a; . spec/assets.env; set +a
# Soft prompt as virtual tokens. SKIP=11 places the 256 tokens at ids 151680..151935 (see pitfall).
.venv/bin/python spec/export_sp.py "$BASE" "$SP_CKPT" spec/qwen3-8b-areal-sp256-off11 11
PHASE=base bash spec/run_spec.sh           # ~1 h each phase
PHASE=sp   bash spec/run_spec.sh
PHASE=lora bash spec/run_spec.sh
.venv/bin/python spec/summarize_spec.py
```

Smoke test (minutes): `NUM=4 MAXNEW=256 PHASE=... bash spec/run_spec.sh`.

## How the soft prompt is served

As in the soft-prompt-distillation repo's evaluation: the 256 learned vectors are written into the input
embedding rows of 256 new tokens (`<|soft_prompt_NNNN|>`), their (untied) LM-head rows are zeroed, and
generation masks them with `logit_bias = -100`. The prompt inserts those ids at the start of the user turn.
To SGLang this is an ordinary checkpoint, so the prefix cache and speculative decoding work unchanged.

## Pitfall: DFlash mask token vs virtual-token ids

`z-lab/Qwen3-8B-DFlash-b16` uses token id **151669** as its block-diffusion `[MASK]`
(`dflash_config.mask_token_id`). The soft-prompt export appends virtual tokens right after the Qwen3
tokenizer, i.e. starting at **151669**, which overwrites the mask embedding and collapses DFlash accept
length to 1.28. Exporting with an offset (`export_sp.py ... 11` → ids 151680–151935) fixes it.
EAGLE-3 does not use that id and is unaffected.
