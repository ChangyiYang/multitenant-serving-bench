# Lightweight closed-loop client for SGLang /generate (streaming), with adapter-selection policies.
# Usage: python client.py --port P --n-adapters N --dist {zipf,uniform,rr,none} --conc C --num R --out f.jsonl --tag T
import argparse, asyncio, json, random, time, os
import aiohttp
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=31000)
ap.add_argument("--n-adapters", type=int, default=0)
ap.add_argument("--dist", default="none")
ap.add_argument("--zipf-s", type=float, default=1.0)
ap.add_argument("--conc", type=int, required=True)
ap.add_argument("--num", type=int, required=True)
ap.add_argument("--warmup", type=int, default=0)
ap.add_argument("--in-len", type=int, default=1024)
ap.add_argument("--out-len", type=int, default=256)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
ap.add_argument("--tag", default="")
ap.add_argument("--load-log", default="")
ap.add_argument("--sp-len", type=int, default=0)
ap.add_argument("--n-tenants", type=int, default=0)
ap.add_argument("--embeds", action="store_true")
a = ap.parse_args()
rng = random.Random(a.seed)
nrng = np.random.default_rng(a.seed)

EMB = None
if a.embeds:
    EMB = (np.random.default_rng(0).standard_normal((a.in_len + a.sp_len, 4096)) * 0.02).round(5).tolist()

def pick(i):
    if a.dist == "none":
        return None
    if a.dist == "rr":
        k = i % a.n_adapters
    elif a.dist == "uniform":
        k = rng.randrange(a.n_adapters)
    elif a.dist == "zipf":
        w = 1.0 / np.arange(1, a.n_adapters + 1) ** a.zipf_s
        k = int(nrng.choice(a.n_adapters, p=w / w.sum()))
    return f"lora{k:03d}"

async def one(sess, i, rec):
    ids = [rng.randrange(1000, 100000) for _ in range(a.in_len)]
    if a.sp_len:
        # Soft prompt modeled as a fixed per-tenant 1024-token prefix (tenant = i mod N, round-robin).
        t = i % a.n_tenants if a.n_tenants else i
        trng = random.Random(10_000_000 + t)
        ids = [trng.randrange(1000, 100000) for _ in range(a.sp_len)] + ids
    payload = {"input_ids": ids,
               "sampling_params": {"max_new_tokens": a.out_len, "temperature": 0, "ignore_eos": True},
               "stream": True}
    if a.embeds:
        payload.pop("input_ids")
        payload["input_embeds"] = EMB[: len(ids)]
    lora = pick(i)
    if lora:
        payload["lora_path"] = lora
    t0 = time.perf_counter(); times = []
    async with sess.post(f"http://127.0.0.1:{a.port}/generate", json=payload) as r:
        r.raise_for_status()
        async for line in r.content:
            line = line.strip()
            if line.startswith(b"data:") and line != b"data: [DONE]":
                times.append(time.perf_counter())
    if rec:
        # Each streamed chunk carries >=1 tokens; ITL is computed per chunk gap.
        itl = np.diff(times) * 1000 if len(times) > 1 else np.array([])
        return {"ttft": (times[0] - t0) * 1000, "e2e": (times[-1] - t0) * 1000,
                "itl": itl.tolist(), "lora": lora}

async def run(n, start, rec):
    out = []; idx = iter(range(start, start + n))
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3600), connector=aiohttp.TCPConnector(limit=0)) as sess:
        async def worker():
            for i in idx:
                r = await one(sess, i, rec)
                if rec:
                    out.append(r)
        t0 = time.perf_counter()
        await asyncio.gather(*[worker() for _ in range(a.conc)])
        return out, time.perf_counter() - t0

def loads():
    try:
        return int(open(a.load_log).read())
    except Exception:
        return None

if a.warmup:
    asyncio.run(run(a.warmup, 10**6, False))
l0 = loads()
res, dur = asyncio.run(run(a.num, 0, True))
l1 = loads()
itl = np.concatenate([np.array(r["itl"]) for r in res if r["itl"]])
ttft = np.array([r["ttft"] for r in res])
summ = {"tag": a.tag, "dist": a.dist, "n_adapters": a.n_adapters, "conc": a.conc, "num": a.num,
        "itl_p50": float(np.median(itl)), "itl_p99": float(np.percentile(itl, 99)),
        "ttft_p50": float(np.median(ttft)), "ttft_p99": float(np.percentile(ttft, 99)),
        "out_tput": a.num * a.out_len / dur, "duration_s": dur,
        "distinct_adapters": len({r["lora"] for r in res}),
        "slot_loads": (l1 - l0) if (l0 is not None and l1 is not None) else None}
with open(a.out, "a") as f:
    f.write(json.dumps(summ) + "\n")
print(json.dumps(summ))
