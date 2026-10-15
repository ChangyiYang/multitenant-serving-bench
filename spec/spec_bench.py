# Measure speculative-decoding accept length on LiveCodeBench v5 (2410-2502) prompts, AReaL/Qwen3 template.
# Optional soft prompt: virtual token ids (exported model) inserted at the start of the user turn.
# Usage: python spec_bench.py --port 31000 --tokenizer <path> --out f.jsonl --tag T [--sp-tokens soft_prompt_tokens.json]
import argparse, asyncio, json, random, time
import aiohttp
from transformers import AutoTokenizer

PLACEHOLDER = "<|soft_prompt|>"

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=31000)
ap.add_argument("--tokenizer", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--sp-tokens", default="")
ap.add_argument("--lora", default="")
ap.add_argument("--num", type=int, default=64)
ap.add_argument("--conc", type=int, default=16)
ap.add_argument("--max-new", type=int, default=8192)
ap.add_argument("--temperature", type=float, default=1.0)
ap.add_argument("--top-p", type=float, default=0.95)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.tokenizer)
sp = json.load(open(a.sp_tokens)) if a.sp_tokens else None
import os
problems = [json.loads(l)["question"] for l in open(os.environ["LCB_PATH"])][: a.num]  # spec/assets.env

def render(q):
    # Matches the run config: model-default system, user = [soft prompt] + problem + "\n/think",
    # generation prompt with enable_thinking, assistant prefix "<think>".
    user = (PLACEHOLDER if sp else "") + q + "\n/think"
    text = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=True) + "<think>"
    if not sp:
        return tok.encode(text, add_special_tokens=False)
    pre, post = text.split(PLACEHOLDER)
    return tok.encode(pre, add_special_tokens=False) + sp["token_ids"] + tok.encode(post, add_special_tokens=False)

def params(i):
    p = {"temperature": a.temperature, "top_p": a.top_p, "max_new_tokens": a.max_new,
         "stop_token_ids": [151645, 151643], "sampling_seed": a.seed * 100003 + i}
    if sp:  # virtual tokens are input-only: never generate them
        p["logit_bias"] = {str(t): sp["logit_bias"] for t in sp["token_ids"]}
    return p

async def one(sess, i, ids):
    t0 = time.perf_counter()
    async with sess.post(f"http://127.0.0.1:{a.port}/generate",
                         json={"input_ids": ids, "sampling_params": params(i),
                               **({"lora_path": a.lora} if a.lora else {})}) as r:
        r.raise_for_status()
        d = await r.json()
    m = d["meta_info"]
    return {"tag": a.tag, "i": i, "prompt_tokens": m.get("prompt_tokens"), "completion_tokens": m.get("completion_tokens"),
            "accept_length": m.get("spec_accept_length"), "accept_rate": m.get("spec_accept_rate"),
            "verify_ct": m.get("spec_verify_ct"), "finish": (m.get("finish_reason") or {}).get("type"),
            "latency_s": time.perf_counter() - t0, "text": d.get("text", "")}

async def main():
    prompts = [render(q) for q in problems]
    sem = asyncio.Semaphore(a.conc)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None),
                                     connector=aiohttp.TCPConnector(limit=0)) as sess:
        async def run(i):
            async with sem:
                return await one(sess, i, prompts[i])
        t0 = time.perf_counter()
        rows = await asyncio.gather(*[run(i) for i in range(len(prompts))])
        dur = time.perf_counter() - t0
    with open(a.out, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    toks = sum(r["completion_tokens"] for r in rows)
    vc = sum(r["verify_ct"] or 0 for r in rows)
    print(json.dumps({"tag": a.tag, "n": len(rows), "out_tokens": toks, "duration_s": round(dur, 1),
                      "out_tok_per_s": round(toks / dur, 1),
                      "accept_length_tokenweighted": round(toks / vc, 3) if vc else None}))

asyncio.run(main())
