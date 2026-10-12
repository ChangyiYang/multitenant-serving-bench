# Run many sglang bench_serving configs in one process (imports paid once).
# Usage: python drive_bench.py <phase>   phase = base | noradix | lora
import os, sys, shlex
from sglang.benchmark import serving

PHASE = sys.argv[1]
MODEL = sys.argv[2]
OUT = "results_real"; os.makedirs(OUT, exist_ok=True)
RATES = [float(x) for x in os.environ.get("RATES", "8 16 24 32 40").split()]
SKIP_MOON = os.environ.get("SKIP_MOON") == "1"
COMMON = f"--backend sglang --port 31000 --model {MODEL} --tokenizer {MODEL} --flush-cache --warmup-requests 4 --disable-tqdm"
QO = "--random-input-len 256 --random-output-len 256 --random-range-ratio 0.5"
LORAS = " ".join(f"lora{i:03d}" for i in range(256))

def gsp(n, ppg, dist):
    z = "--gsp-group-distribution zipf --gsp-zipf-alpha 1.0" if dist == "zipf" else ""
    return (f"--dataset-name generated-shared-prefix --gsp-num-groups {n} --gsp-prompts-per-group {ppg} "
            f"--gsp-system-prompt-len 1024 --gsp-question-len 256 --gsp-output-len 256 --gsp-range-ratio 0.5 {z}")

def run(tag, rate, extra, env=None):
    os.environ.pop("LORA_TRUE_ZIPF", None)
    if env:
        os.environ.update(env)
    sys.argv = ["bench"] + shlex.split(f"{COMMON} --request-rate {rate} --output-file {OUT}/{tag}.jsonl --tag {tag} {extra}")
    print(f"=== {tag} rate={rate}", flush=True)
    try:
        serving.cli_main()
    except Exception as e:  # keep the sweep going; record the failure
        print(f"FAILED {tag} rate={rate}: {e!r}", flush=True)

MOON = "--dataset-name mooncake --mooncake-workload mooncake --use-trace-timestamps --mooncake-slowdown-factor 2 --num-prompts 1000"

if PHASE == "base":
    for r in RATES:
        run("base_random", r, f"--dataset-name random --num-prompts 1024 {QO}")
    for r in RATES:
        run("sp_cache_N256_uniform", r, gsp(256, 4, "uniform"))
    for r in RATES:
        run("sp_cache_N256_zipf", r, gsp(256, 4, "zipf"))
    for r in RATES:
        run("sp_cache_N1024_zipf", r, gsp(1024, 1, "zipf"))
    if not SKIP_MOON: run("mooncake_base", "inf", MOON)
elif PHASE == "noradix":
    for r in RATES:
        run("sp_nocache_N256_uniform", r, gsp(256, 4, "uniform"))
elif PHASE == "lora":
    for r in RATES:
        run("lora256_uniform", r, f"--dataset-name random --num-prompts 1024 {QO} --lora-name {LORAS} --lora-request-distribution uniform")
    for r in RATES:
        run("lora256_zipf", r, f"--dataset-name random --num-prompts 1024 {QO} --lora-name {LORAS} "
            f"--lora-request-distribution skewed --lora-zipf-alpha 1.01", env={"LORA_TRUE_ZIPF": "1.0"})
    if not SKIP_MOON: run("mooncake_lora256_zipf", "inf", f"{MOON} --lora-name {LORAS} --lora-request-distribution skewed --lora-zipf-alpha 1.01",
        env={"LORA_TRUE_ZIPF": "1.0"})
print("PHASE_DONE", PHASE, flush=True)
