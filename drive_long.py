# Long-context SP vs baseline vs LoRA: 32K input, 1024 output, SP = 256 tokens/tenant. One process per server.
# Usage: python drive_long.py <phase> <model>   phase = base | noradix | lora
import os, sys, shlex
from sglang.benchmark import serving

PHASE, MODEL = sys.argv[1], sys.argv[2]
OUT = os.environ.get("OUT", "results_long"); os.makedirs(OUT, exist_ok=True)
RATES = [float(x) for x in os.environ.get("RATES", "0.5 1 1.5 2 3").split()]
COMMON = f"--backend sglang --port 31000 --model {MODEL} --tokenizer {MODEL} --flush-cache --warmup-requests 2 --disable-tqdm"
N = int(os.environ.get("NREQ", "128"))
RAND = f"--dataset-name random --num-prompts {N} --random-input-len 32000 --random-output-len 1024 --random-range-ratio 1.0"
# gsp decodes random vocab ids then re-tokenizes (~5.3% inflation), so lengths are pre-scaled via env to land at 256 + 32000.
GSP = (f"--dataset-name generated-shared-prefix --gsp-num-groups {N} --gsp-prompts-per-group 1 --gsp-system-prompt-len {os.environ.get('SPLEN', '256')} "
       f"--gsp-question-len {os.environ.get('QLEN', '32000')} --gsp-output-len 1024 --gsp-range-ratio 1.0 --gsp-group-distribution zipf --gsp-zipf-alpha 1.0")
LORAS = " ".join(f"lora{i:03d}" for i in range(128))

def run(tag, rate, extra, env=None):
    os.environ.pop("LORA_TRUE_ZIPF", None)
    if env:
        os.environ.update(env)
    sys.argv = ["bench"] + shlex.split(f"{COMMON} --request-rate {rate} --output-file {OUT}/{tag}.jsonl --tag {tag} {extra}")
    print(f"=== {tag} rate={rate}", flush=True)
    try:
        serving.cli_main()
    except Exception as e:
        print(f"FAILED {tag} rate={rate}: {e!r}", flush=True)

if PHASE == "base":
    for r in RATES:
        run("long_base", r, RAND)
    for r in RATES:
        run("long_sp_cache", r, GSP)
elif PHASE == "noradix":
    for r in RATES:
        run("long_sp_nocache", r, GSP)
elif PHASE == "lora":
    for r in RATES:
        run("long_lora128_zipf", r, f"{RAND} --lora-name {LORAS} --lora-request-distribution skewed --lora-zipf-alpha 1.01",
            env={"LORA_TRUE_ZIPF": "1.0"})
print("PHASE_DONE", PHASE, flush=True)
