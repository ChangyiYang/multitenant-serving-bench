import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, INK2, MUTED, GRID, BASE, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
# Categorical slots in fixed order; texture of identity also carried by marker + linestyle + end labels.
ARMS = [("long_base", "Baseline (no SP, no LoRA)", MUTED, "--", "o"),
        ("long_sp_cache", "SP 256 tok + prefix cache, 128 tenants, Zipf", "#2a78d6", "-", "o"),
        ("long_sp_nocache", "SP 256 tok, no prefix cache (SGLang today)", "#eb6834", "-", "s"),
        ("long_lora128_zipf", "LoRA r16, 128 adapters in GPU, Zipf", "#1baf7a", "-", "^")]
SHORT = {"long_base": "Baseline", "long_sp_cache": "", "long_sp_nocache": "SP (cache or not)", "long_lora128_zipf": "LoRA 128"}

def load(tag):
    f = f"results_long/{tag}.jsonl"
    if not os.path.exists(f):
        return []
    rows = [json.loads(l) for l in open(f)]
    return sorted(rows, key=lambda d: d["request_rate"])

plt.rcParams.update({"font.size": 10.5, "axes.edgecolor": BASE, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
fig, ax = plt.subplots(1, 3, figsize=(20, 6.2), facecolor=SURF)
panels = [("median_itl_ms", "A. Per-token latency (ITL p50, ms)", 1),
          ("median_ttft_ms", "B. Time to first token (TTFT p50, ms, log scale)", 1),
          ("output_throughput", "C. Achieved output throughput (tok/s)", 1)]
for a, (key, title, scale) in zip(ax, panels):
    a.set_facecolor(SURF)
    for tag, label, color, ls, mk in ARMS:
        rows = [d for d in load(tag) if d["request_rate"] != "trace"]
        if not rows:
            continue
        xs = [d["request_rate"] for d in rows]; ys = [d[key] * scale for d in rows]
        a.plot(xs, ys, ls, color=color, lw=2, marker=mk, ms=7, markeredgecolor=SURF, markeredgewidth=2, label=label, zorder=3)
        if key != "median_itl_ms" and tag != "long_sp_cache":
            a.annotate(SHORT[tag], (xs[-1], ys[-1]), xytext=(6, 8 if tag == "long_base" else (-6 if tag == "long_sp_nocache" else 0)), textcoords="offset points", va="center", fontsize=9, color=INK2)
    a.set_title(title, loc="left", fontsize=11.5, color=INK)
    a.set_xlabel("Offered load (requests/s, Poisson arrivals)"); a.set_xticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    a.set_xlim(0.25, 0.92)
    if key == "median_ttft_ms":
        a.set_yscale("log")
    else:
        a.set_ylim(bottom=0)
    a.grid(axis="y", color=GRID, lw=0.8); a.set_axisbelow(True)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
ax[2].legend(frameon=False, loc="lower right", labelcolor=INK2, fontsize=9.5)
fig.suptitle("Long context (sglang bench_serving, Poisson): soft prompt (256 tok/tenant) vs LoRA r16 - Qwen3-8B, 1x B300, "
             "per-request input 32K / output 1024", x=0.01, ha="left", fontsize=12.5, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig("long_context.png", dpi=140, facecolor=SURF)
print("saved")
