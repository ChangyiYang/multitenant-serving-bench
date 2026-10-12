# Multi-tenant serving on SGLang: multi-LoRA vs soft prompts

Benchmarks run Oct 7–12 2026 on **1× NVIDIA B300** (Slurm interactive allocation, 16 CPU, 200 GB RAM) with
**Qwen3-8B (bf16)**, **SGLang 0.5.21** (torch 2.13.0+cu130, flashinfer 0.6.18, Triton 3.7.1). Exact environment: `requirements-lock.txt`.

## Setup

```bash
./setup.sh                                  # venv, sglang 0.5.21, bench patch, 1024 synthetic r16 adapters (~86 GB)
export MODEL=/path/to/Qwen3-8B              # or Qwen/Qwen3-8B
```

All scripts must run **on a GPU node** (inside the interactive allocation), from the repo root. Each script starts its own
SGLang server on port 31000 and writes into a `results*/` directory. Re-render figures with `.venv/bin/python plot_*.py`.

## Experiments

| # | Question | Command | Results | Figure |
|---|---|---|---|---|
| 1 | Multi-LoRA, all adapters in GPU, 8–256 adapters | `for a in nolora 8 32 128 256; do bash run_bench.sh $a; done` | `results/` | `figures/multilora_qwen3_8b.png` |
| 2 | 512 / 1024 adapters; running-request cap 1024 | `MRR=1024 TAG=_mrr1024 CONCS="1 32 256 512 1024" bash run_bench.sh {nolora,512,1024}`; control: `MRR=1024 TAG=_mrr1024 CONCS="1 32 256" bash run_bench.sh 256` | `results/*_mrr1024` | `figures/multilora_summary.png` (A–C) |
| 3 | LoRA CPU offload: 32 GPU slots, N adapters in CPU | `for x in 32 64 "64 overlap" 128 256 1024; do bash run_offload.sh $x; done` (C=32); C=128: `OROOT=results_offload_fix OCONCS=128 bash run_offload.sh …` | `results_offload/`, `results_offload_fix/` | `figures/multilora_summary.png` (D–F) |
| 4 | Soft prompt (1024 tok) vs LoRA, fixed concurrency | `bash run_sp.sh` | `results_sp/` | `figures/sp_vs_lora.png` |
| 5 | Realistic traffic (Poisson, Zipf tenants, Mooncake trace) | `bash run_real.sh`; high load: `RATES="56 72 88" SKIP_MOON=1 bash run_real.sh` | `results_real/` | `figures/real_traffic.png` |
| 6 | Long context: 32K input, 256-token soft prompt | `RATES="0.3 0.4 0.5 0.6 0.7 0.8" NREQ=64 SPLEN=243 QLEN=30390 bash run_long.sh` | `results_long/` | `figures/long_context.png` |

Common workload unless stated: input 1024 / output 256 tokens, `ignore_eos`, rank-16 all-linear synthetic adapters
(random weights — measures overhead, not quality), `--mem-fraction-static 0.80`, `--max-running-requests 256`,
`--cuda-graph-max-bs-decode 256`, LoRA backend `csgmv` (default).

## Key results

- **LoRA fixed cost**: enabling LoRA adds ~6 ms per decode step on Qwen3-8B (3.55 → 9.5 ms at C=1), independent of rank and adapter count.
- **Capacity**: 1024 r16 adapters fit resident on one B300 (KV pool 1.44M → 0.84M tokens). 256 slots cost ~10% of KV.
- **The real trap is `--max-running-requests` / CUDA-graph capture size**: 256 → 1024 doubles ITL (9.5 → 19.5 ms) regardless of adapter count (csgmv launches by capacity).
- **CPU offload** (32 GPU slots): per-token speed unchanged; cost is slot queueing — at C=128, N=1024 adapters: 1.8–2.8K tok/s, TTFT 6.5–15 s vs ~5.2K tok/s / 0.8 s all-resident. `--enable-lora-overlap-loading` helps only 3–10%.
- **Soft prompt (SP) with prefix cache beats LoRA** under realistic traffic: sustained ~13K vs ~7.2K tok/s (256 tenants), latency close to baseline.
- **SP cost scales with soft-token share of context**: 256 SP tokens + 32K input ≈ baseline (within 1%); 1024 SP tokens + short input costs decode KV bandwidth at high concurrency.
- **SGLang SP today** (`input_embeds`) is impractical: JSON float arrays (8.3 s TTFT at C=1, broken pipe at C≥8) and it forces `--disable-radix-cache`.

## Gotchas found along the way

- `aiohttp.ClientSession` defaults to **100 connections** — set `TCPConnector(limit=0)` (fixed in `client.py`); `results_sp_conn100/` and the C=128 points in `results_offload/` were capped at ~100 and are superseded by `results_sp/` and `results_offload_fix/`.
- bench_serving's `generated-shared-prefix` dataset decodes random vocab ids and re-tokenizes, inflating lengths by ~5%: verify `total_input_tokens` (`results_long_v1/` is the inflated first pass; `SPLEN=243 QLEN=30390` corrects it).
- bench_serving's LoRA `skewed` distribution is geometric (`alpha^-i`), not Zipf: `patches/bench_serving_true_zipf_lora.patch` adds `LORA_TRUE_ZIPF=<s>`.
- `patches/sitecustomize_dir/sitecustomize.py` counts LoRA CPU→GPU slot loads (enabled by `LORA_LOAD_COUNTER=1`, used by `run_offload.sh`).
- Importing sglang from a network filesystem costs ~3 min per `bench_serving` invocation; `drive_*.py` run many configs in one process.
- The "SP + prefix cache" arms model the soft prompt as a fixed per-tenant token-id prefix so the radix cache can match it — i.e. they assume a framework that keys the prefix cache by tenant.
