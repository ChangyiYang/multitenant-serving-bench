import json, re, glob, os, statistics as st

ARMS = ["nolora", "8", "32", "128", "256"]
rows = {}
for arm in ARMS:
    f = f"results/{arm}/bench.jsonl"
    if not os.path.exists(f):
        continue
    for line in open(f):
        d = json.loads(line)
        rows.setdefault(arm, {}).setdefault(d["max_concurrency"], []).append(d)
    log = open(f"results/{arm}/server.log").read()
    m = re.search(r"#tokens: (\d+)", log)
    rows[arm]["kv_tokens"] = int(m.group(1)) if m else None

print("arm | C | ITL p50 ms | TTFT p50 ms | out tok/s | KV tokens")
for arm in ARMS:
    if arm not in rows:
        continue
    for c in sorted(k for k in rows[arm] if isinstance(k, int)):
        ds = rows[arm][c]
        itl = st.mean(d["median_itl_ms"] for d in ds)
        ttft = st.mean(d["median_ttft_ms"] for d in ds)
        tput = st.mean(d["output_throughput"] for d in ds)
        print(f"{arm} | {c} | {itl:.2f} | {ttft:.0f} | {tput:.0f} | {rows[arm]['kv_tokens']}")
