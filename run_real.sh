#!/bin/bash
# Realistic-traffic SP vs LoRA comparison using sglang bench_serving (Poisson arrivals, gsp/random/mooncake).
set -u
cd "$(dirname "$0")"; . .venv/bin/activate
[ -f mooncake_trace.jsonl ] || curl -sfL -o mooncake_trace.jsonl https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/arxiv-trace/mooncake_trace.jsonl
cp mooncake_trace.jsonl /tmp/mooncake_trace.jsonl
MODEL=${MODEL:?set MODEL to a local Qwen3-8B path or HF id}
PORT=31000; mkdir -p results_real
start() {  # name args...
  local name=$1; shift
  python -m sglang.launch_server --model-path $MODEL --port $PORT --mem-fraction-static 0.80 \
    --max-running-requests 256 --cuda-graph-max-bs-decode 256 --random-seed 0 "$@" > results_real/server_$name.log 2>&1 &
  SPID=$!
  until curl -sf localhost:$PORT/health >/dev/null; do sleep 5; kill -0 $SPID 2>/dev/null || { echo "server died $name"; return 1; }; done
  echo "server up $name"
}
stop() { kill $SPID; wait $SPID 2>/dev/null; }
start base && { python drive_bench.py base $MODEL; stop; }
start noradix --disable-radix-cache && { python drive_bench.py noradix $MODEL; stop; }
PATHS=""; for i in $(seq 0 255); do n=$(printf lora%03d $i); PATHS="$PATHS $n=adapters_r16/$n"; done
start lora --enable-lora --max-lora-rank 16 --max-loras-per-batch 256 --max-loaded-loras 256 --lora-paths $PATHS && { python drive_bench.py lora $MODEL; stop; }
echo REAL_DONE
