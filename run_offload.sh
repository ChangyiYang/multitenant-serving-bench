#!/bin/bash
# CPU-offload experiment: S GPU slots, N adapters resident in CPU (N >= S), traffic triggers slot loads/evictions.
# Usage (inside allocation): bash run_offload.sh <N> [overlap]
set -u
cd "$(dirname "$0")"; . .venv/bin/activate

MODEL=${MODEL:?set MODEL to a local Qwen3-8B path or HF id}
N=$1; MODE=${2:-sync}; S=${S:-32}; R=16; PORT=31000
OUT=${OROOT:-results_offload}/N${N}_S${S}_${MODE}; mkdir -p $OUT
PATHS=""
for i in $(seq 0 $((N-1))); do n=$(printf lora%03d $i); PATHS="$PATHS $n=adapters_r$R/$n"; done
EXTRA=""; [ "$MODE" = "overlap" ] && EXTRA="--enable-lora-overlap-loading"
export LORA_LOAD_COUNTER=1 LORA_LOAD_LOG=$PWD/$OUT/slot_loads.txt PYTHONPATH=$PWD/patches/sitecustomize_dir
python -m sglang.launch_server --model-path $MODEL --port $PORT --mem-fraction-static 0.80 \
  --max-running-requests 256 --cuda-graph-max-bs-decode 256 --random-seed 0 \
  --enable-lora --max-lora-rank $R --max-loras-per-batch $S --max-loaded-loras $N $EXTRA \
  --lora-paths $PATHS > $OUT/server.log 2>&1 &
SPID=$!
unset PYTHONPATH LORA_LOAD_COUNTER
until curl -sf localhost:$PORT/health >/dev/null; do
  sleep 5; kill -0 $SPID 2>/dev/null || { echo "server died N=$N"; tail -30 $OUT/server.log; exit 1; }
done
echo "server up N=$N mode=$MODE"
for C in ${OCONCS:-32 128}; do
  for D in zipf uniform rr; do
    NUM=$(( C*5 )); [ $NUM -lt 64 ] && NUM=64
    python client.py --port $PORT --n-adapters $N --dist $D --conc $C --num $NUM --warmup $C \
      --out $OUT/results.jsonl --tag "N${N}_S${S}_${MODE}" --load-log $OUT/slot_loads.txt > $OUT/c${C}_${D}.log 2>&1
    echo "N=$N C=$C D=$D exit=$?"
  done
done
kill $SPID; wait $SPID 2>/dev/null
echo "OFFLOAD_DONE N=$N $MODE"
