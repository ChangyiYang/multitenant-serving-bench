"""Accept-length / throughput table from spec/results/*.jsonl(.gz).
Usage: python spec/summarize_spec.py [results_dir]"""
import gzip, json, os, sys
from collections import defaultdict

d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "results")
rows = defaultdict(list)
for fn in sorted(os.listdir(d)):
    if fn.startswith("run_summaries") or not (fn.endswith(".jsonl") or fn.endswith(".jsonl.gz")):
        continue
    op = gzip.open if fn.endswith(".gz") else open
    for line in op(os.path.join(d, fn), "rt"):
        r = json.loads(line)
        rows[r["tag"]].append(r)

print(f"{'tag':40} {'n':>3} {'out_tok':>8} {'accept_len':>10} {'mean_latency_s':>14}")
for tag, rs in rows.items():
    toks = sum(r["completion_tokens"] for r in rs)
    vc = sum(r["verify_ct"] or 0 for r in rs)
    al = f"{toks / vc:.3f}" if vc else "-"
    lat = sum(r["latency_s"] for r in rs) / len(rs)
    print(f"{tag:40} {len(rs):>3} {toks:>8} {al:>10} {lat:>14.1f}")
