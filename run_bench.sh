#!/bin/bash
# Qwen3-8B multi-LoRA sweep, no CPU offload: #loaded adapters == --max-loras-per-batch == --max-loaded-loras.
# Usage (inside the interactive allocation): bash run_bench.sh <arm> ; arm = nolora | 8 | 32 | 128 | 256
set -u
cd "$(dirname "$0")"; . .venv/bin/activate

MODEL=${MODEL:?set MODEL to a local Qwen3-8B path or HF id}
ARM=$1; PORT=31000; R=16; MRR=${MRR:-256}; TAG=${TAG:-}
OUT=results/$ARM$TAG; mkdir -p $OUT
CONCS=${CONCS:-"1 8 32 128 256"}

COMMON="--model-path $MODEL --port $PORT --mem-fraction-static 0.80 --max-running-requests $MRR --cuda-graph-max-bs-decode $MRR --random-seed 0"
LORA_ARGS=""; NAMES=""
if [ "$ARM" != "nolora" ]; then
  N=$ARM; PATHS=""
  for i in $(seq 0 $((N-1))); do n=$(printf lora%03d $i); PATHS="$PATHS $n=adapters_r$R/$n"; NAMES="$NAMES $n"; done
  LORA_ARGS="--enable-lora --max-lora-rank $R --max-loras-per-batch $N --max-loaded-loras $N --lora-paths $PATHS"
fi

python -m sglang.launch_server $COMMON $LORA_ARGS > $OUT/server.log 2>&1 &
SPID=$!
until curl -sf localhost:$PORT/health >/dev/null; do
  sleep 5; kill -0 $SPID 2>/dev/null || { echo "server died"; tail -30 $OUT/server.log; exit 1; }
done
echo "server up"; nvidia-smi --query-gpu=memory.used --format=csv,noheader > $OUT/gpu_mem.txt

for C in $CONCS; do
  NP=$(( C*5 > 16 ? C*5 : 16 ))
  LORA_BENCH=""; [ -n "$NAMES" ] && LORA_BENCH="--lora-name $NAMES --lora-request-distribution distinct"
  for rep in 1 2; do
    python -m sglang.bench_serving --backend sglang --port $PORT --dataset-name random \
      --random-input-len 1024 --random-output-len 256 --random-range-ratio 1.0 \
      --num-prompts $NP --max-concurrency $C --seed $rep --warmup-requests $(( C < 8 ? 8 : C )) \
      $LORA_BENCH --output-file $OUT/bench.jsonl > $OUT/bench_c${C}_r${rep}.log 2>&1
    echo "C=$C rep=$rep exit=$?"
  done
done
kill $SPID; wait $SPID 2>/dev/null
echo "ARM_DONE $ARM"
