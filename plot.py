import json, os, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARMS = [("nolora", "LoRA off", "#898781", "--"), ("8", "8 adapters", "#86b6ef", "-"),
        ("32", "32 adapters", "#3987e5", "-"), ("128", "128 adapters", "#1c5cab", "-"),
        ("256", "256 adapters", "#0d366b", "-")]
INK, MUTED, GRID, BASE, SURF = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

def load(arm):
    f = f"results/{arm}/bench.jsonl"
    by = {}
    if os.path.exists(f):
        for line in open(f):
            d = json.loads(line)
            by.setdefault(d["max_concurrency"], []).append(d)
    cs = sorted(by)
    return cs, [st.mean(x["median_itl_ms"] for x in by[c]) for c in cs], \
        [st.mean(x["output_throughput"] for x in by[c]) for c in cs]

plt.rcParams.update({"font.size": 11, "axes.edgecolor": BASE, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=SURF)
panels = [(1, "Per-token latency (ITL p50, ms) - lower is better"),
          (2, "Output throughput (tok/s) - higher is better")]
for ax, (k, title) in zip(axes, panels):
    ax.set_facecolor(SURF)
    for arm, label, color, ls in ARMS:
        cs, itl, tput = load(arm)
        if not cs:
            continue
        ys = itl if k == 1 else tput
        ax.plot(cs, ys, ls, color=color, lw=2, marker="o", ms=7,
                markeredgecolor=SURF, markeredgewidth=2, label=label, zorder=3)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 8, 32, 128, 256]); ax.set_xticklabels(["1", "8", "32", "128", "256"])
    ax.set_xlabel("Concurrent requests (each request uses a different adapter, round-robin)")
    ax.set_title(title, loc="left", fontsize=12, color=INK)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].legend(frameon=False, loc="upper left", labelcolor="#52514e")
fig.suptitle("SGLang 0.5.21 multi-LoRA, Qwen3-8B, 1x B300, rank-16 adapters, no CPU offload "
             "(loaded adapters = GPU slots), input 1024 / output 256",
             x=0.01, ha="left", fontsize=12, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig("multilora_qwen3_8b.png", dpi=150, facecolor=SURF)
print("saved")
