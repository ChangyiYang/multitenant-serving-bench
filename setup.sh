#!/bin/bash
# Create the venv used by every script (SGLang 0.5.21, torch 2.13 + cu130), apply the bench patch, generate adapters.
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv && . .venv/bin/activate
pip install -q uv
uv pip install "sglang[all]==0.5.21" matplotlib   # exact environment: requirements-lock.txt
# True-Zipf LoRA assignment for bench_serving (needed by run_real.sh / run_long.sh LoRA arms).
SITE=$(python -c "import sglang, os; print(os.path.dirname(os.path.dirname(sglang.__file__)))")
patch -d "$SITE" -p2 < patches/bench_serving_true_zipf_lora.patch
# Synthetic rank-16 all-linear Qwen3-8B adapters (random weights; ~84 MB each, ~86 GB for 1024).
python make_adapters.py ${N_ADAPTERS:-1024} 16 adapters_r16
