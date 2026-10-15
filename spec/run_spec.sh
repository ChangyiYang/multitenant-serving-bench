#!/bin/bash
# Speculative-decoding accept length on Qwen3-8B with base / soft prompt / LoRA targets.
# Prereqs (see spec/README.md): ./setup.sh, python spec/fetch_assets.py, and for PHASE=sp
# the exported soft-prompt model (spec/export_sp.py). Run on a GPU node from the repo root:
#   PHASE=base bash spec/run_spec.sh     # then PHASE=sp, PHASE=lora
# Env knobs: NUM (prompts at concurrency 16, default 64), MAXNEW (default 8192), ONLY=1 (DFlash only).
set -u
cd "$(dirname "$0")/.."; . .venv/bin/activate
set -a; . spec/assets.env; set +a
PHASE=${PHASE:-base}
SPMODEL=${SPMODEL:-spec/qwen3-8b-areal-sp256-off11}
LORA_SRV=""; SPARG=""; MODEL=$BASE
if [ "$PHASE" = sp ]; then
  MODEL=$SPMODEL; SPARG="--sp-tokens $MODEL/soft_prompt_tokens.json"
elif [ "$PHASE" = lora ]; then
  SPARG="--lora areal8"
  LORA_SRV="--enable-lora --lora-paths areal8=$LORA_DIR --max-lora-rank 1 --max-loras-per-batch 2"
fi
OUT=spec/results; mkdir -p $OUT; PORT=${PORT:-31000}
NUM=${NUM:-64}; MAXNEW=${MAXNEW:-8192}
COMMON="--model-path $MODEL --port $PORT --mem-fraction-static 0.75 --max-running-requests 64 --random-seed 0 --context-length 40960"

arm() {  # name extra-server-args...
  local name=$1; shift
  python -m sglang.launch_server $COMMON $LORA_SRV "$@" > $OUT/server_${PHASE}_$name.log 2>&1 &
  local pid=$!
  until curl -sf localhost:$PORT/health >/dev/null; do
    sleep 5; kill -0 $pid 2>/dev/null || { echo "server died $name"; tail -20 $OUT/server_${PHASE}_$name.log; return 1; }
  done
  for C in 1 16; do
    python spec/spec_bench.py --port $PORT --tokenizer $MODEL --out $OUT/${PHASE}.jsonl --tag "${PHASE}_${name}_c$C" \
      --num $([ $C = 1 ] && echo $(( NUM < 16 ? NUM : 16 )) || echo $NUM) --conc $C --max-new $MAXNEW $SPARG
  done
  kill $pid; wait $pid 2>/dev/null
}
[ -z "${ONLY:-}" ] && arm nospec
[ -z "${ONLY:-}" ] && arm eagle3_tengyun_chain  --speculative-algorithm EAGLE3 --speculative-draft-model-path $EAGLE_T --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
[ -z "${ONLY:-}" ] && arm eagle3_angelslim_chain --speculative-algorithm EAGLE3 --speculative-draft-model-path $EAGLE_A --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
arm dflash_b16 --speculative-algorithm DFLASH --speculative-draft-model-path $DFLASH
echo "SPEC_DONE $PHASE"
