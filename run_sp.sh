#!/bin/bash
# Soft-prompt (SP) serving curves for Qwen3-8B, mirroring the LoRA sweep. Run inside the interactive allocation.
# Phase 1 (radix ON):  A = no SP baseline; C = per-tenant fixed 1024-token prefix, N tenants (ideal multi-SP w/ prefix cache)
# Phase 2 (radix OFF): B = 1024 extra tokens per request (SGLang input_embeds compute path); D = real input_embeds JSON path
set -u
cd "$(dirname "$0")"; . .venv/bin/activate

MODEL=${MODEL:?set MODEL to a local Qwen3-8B path or HF id}
PORT=31000; OUT=results_sp; mkdir -p $OUT
CONCS="1 8 32 128 256"

start() {
  python -m sglang.launch_server --model-path $MODEL --port $PORT --mem-fraction-static 0.80 \
    --max-running-requests 256 --cuda-graph-max-bs-decode 256 --random-seed 0 "$@" > $OUT/server_$TAGS.log 2>&1 &
  SPID=$!
  until curl -sf localhost:$PORT/health >/dev/null; do
    sleep 5; kill -0 $SPID 2>/dev/null || { echo "server died $TAGS"; tail -30 $OUT/server_$TAGS.log; exit 1; }
  done
  echo "server up $TAGS"
}
run() {  # tag conc extra-client-args...
  local tag=$1 C=$2; shift 2
  local NUM=$(( C*5 > 16 ? C*5 : 16 ))
  python client.py --port $PORT --conc $C --num $NUM --out $OUT/results.jsonl --tag $tag "$@" > $OUT/${tag}_c${C}.log 2>&1
  echo "$tag C=$C exit=$?"
}
flush() { curl -s -X POST localhost:$PORT/flush_cache >/dev/null; sleep 2; }

TAGS=radix; start
for C in $CONCS; do flush; run A_nosp $C --warmup $C; done
for N in 8 32 128 256 1024; do
  for C in $CONCS; do
    flush; W=$(( N > C ? N : C ))
    run C_sp_N$N $C --sp-len 1024 --n-tenants $N --warmup $W
  done
done
kill $SPID; wait $SPID 2>/dev/null

TAGS=noradix; start --disable-radix-cache
for C in $CONCS; do run B_sp_nocache $C --sp-len 1024 --warmup $C; done
# D (input_embeds) measured in results_sp_conn100: C=1 TTFT 8.3 s, C>=8 broken pipe
kill $SPID; wait $SPID 2>/dev/null
echo "SP_DONE"
