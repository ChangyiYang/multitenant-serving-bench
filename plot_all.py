import json, os, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, INK2, MUTED, GRID, BASE, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SEQ = {"8": "#86b6ef", "32": "#3987e5", "128": "#1c5cab", "256": "#0d366b"}
CAT = {"zipf": ("#2a78d6", "o", "Zipf hot set"), "uniform": ("#eb6834", "s", "Uniform random"),
       "rr": ("#1baf7a", "^", "Round-robin (worst case)")}
X = [1, 8, 32, 128, 256, 512, 1024]

def bench(arm):
    f = f"results/{arm}/bench.jsonl"; by = {}
    if os.path.exists(f):
        for line in open(f):
            d = json.loads(line); by.setdefault(d["max_concurrency"], []).append(d)
    cs = sorted(by)
    return (cs, [st.mean(x["median_itl_ms"] for x in by[c]) for c in cs],
            [st.mean(x["output_throughput"] for x in by[c]) for c in cs],
            [st.mean(x["median_ttft_ms"] for x in by[c]) for c in cs])

def offload():
    rows = []
    for n in (32, 64, 128, 256, 1024):
        # C=32 from the original run; C=128 from the rerun with the 100-connection client cap removed
        for root, c in (("results_offload", 32), ("results_offload_fix", 128)):
            f = f"{root}/N{n}_S32_sync/results.jsonl"
            if os.path.exists(f):
                rows += [r for r in map(json.loads, open(f)) if r["conc"] == c]
    return rows

def style(ax, title, xlabel, ylabel, logx=True, ticks=X):
    ax.set_facecolor(SURF)
    if logx:
        ax.set_xscale("log", base=2); ax.set_xticks(ticks); ax.set_xticklabels([str(t) for t in ticks])
        ax.minorticks_off()
    ax.set_title(title, loc="left", fontsize=11.5, color=INK)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_ylim(bottom=0)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

def line(ax, xs, ys, color, label, ls="-", marker="o"):
    ax.plot(xs, ys, ls, color=color, lw=2, marker=marker, ms=7, markeredgecolor=SURF,
            markeredgewidth=2, label=label, zorder=3)

plt.rcParams.update({"font.size": 10.5, "axes.edgecolor": BASE, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
fig, ax = plt.subplots(2, 3, figsize=(19, 10.5), facecolor=SURF)

# A/B: all adapters resident in GPU, server cap 256
for arm, label, color, ls in [("nolora", "LoRA off", MUTED, "--")] + [(k, f"{k} adapters", v, "-") for k, v in SEQ.items()]:
    cs, itl, tput, _ = bench(arm)
    line(ax[0, 0], cs, itl, color, label, ls); line(ax[0, 1], cs, tput, color, label, ls)
style(ax[0, 0], "A. All adapters in GPU: per-token latency (ms)", "Concurrent requests", "ITL p50 (ms)", ticks=X[:5])
style(ax[0, 1], "B. All adapters in GPU: output throughput (tok/s)", "Concurrent requests", "tok/s", ticks=X[:5])
ax[0, 0].legend(frameon=False, loc="upper left", labelcolor=INK2)

# C: server request cap (= CUDA-graph capture size) drives the fixed cost
for arm, label, color, ls in [("nolora_mrr1024", "LoRA off, cap 1024", MUTED, "--"),
                              ("256", "256 adapters, cap 256", "#86b6ef", "-"),
                              ("256_mrr1024", "256 adapters, cap 1024", "#3987e5", "-"),
                              ("512_mrr1024", "512 adapters, cap 1024", "#1c5cab", "-"),
                              ("1024_mrr1024", "1024 adapters, cap 1024", "#0d366b", "-")]:
    cs, itl, _, _ = bench(arm)
    line(ax[0, 2], cs, itl, color, label, ls)
style(ax[0, 2], "C. Raising the running-request cap 256 -> 1024 doubles latency", "Concurrent requests", "ITL p50 (ms)")
ax[0, 2].legend(frameon=False, loc="upper left", labelcolor=INK2)

# D/E/F: CPU offload, 32 GPU slots, N adapters in CPU
rows = offload()
_, _, res_tput, res_ttft = bench("128")  # resident reference at C=128 (128 slots)
for d, (color, mk, label) in CAT.items():
    for k, (c, key, a, scale) in enumerate([(128, "out_tput", ax[1, 0], 1), (128, "ttft_p50", ax[1, 1], 1e-3),
                                            (32, "out_tput", ax[1, 2], 1)]):
        pts = sorted((r["n_adapters"], r[key] * scale) for r in rows if r["dist"] == d and r["conc"] == c)
        if pts:
            xs, ys = zip(*pts)
            line(a, xs, ys, color, label, marker=mk)
            a.annotate(label.split(" (")[0], (xs[-1], ys[-1]), xytext=(6, {"zipf": 0, "uniform": 7, "rr": -7}[d]), textcoords="offset points",
                       va="center", fontsize=9, color=INK2)
xt = [32, 64, 128, 256, 1024]
style(ax[1, 0], "D. CPU offload, 32 GPU slots, 128 concurrent: throughput", "Adapters held in CPU memory (N)", "tok/s", ticks=xt)
style(ax[1, 1], "E. CPU offload, 128 concurrent: time to first token (s)", "Adapters held in CPU memory (N)", "TTFT p50 (s)", ticks=xt)
style(ax[1, 2], "F. CPU offload, 32 concurrent: throughput", "Adapters held in CPU memory (N)", "tok/s", ticks=xt)
ref_t = dict(zip(bench("128")[0], bench("128")[2])); ref_f = dict(zip(bench("128")[0], bench("128")[3]))
if 128 in ref_t:
    ax[1, 0].axhline(ref_t[128], color=MUTED, ls="--", lw=1.2)
    ax[1, 0].annotate("128 adapters all in GPU", (32, ref_t[128]), xytext=(0, 5), textcoords="offset points", fontsize=9, color=INK2)
    ax[1, 1].axhline(ref_f[128] / 1000, color=MUTED, ls="--", lw=1.2)
    ax[1, 1].annotate("128 adapters all in GPU", (32, ref_f[128] / 1000), xytext=(0, 5), textcoords="offset points", fontsize=9, color=INK2)
for a in ax[1]:
    a.set_xlim(right=2600)
ax[1, 0].set_ylim(0, 6300)
ax[0, 2].annotate("256 / 512 / 1024 adapters at cap 1024 overlap", (1, 19.5), xytext=(0, 8), textcoords="offset points", fontsize=9, color=INK2)
ax[1, 0].legend(frameon=False, loc="upper right", labelcolor=INK2)

fig.suptitle("SGLang 0.5.21 multi-LoRA on 1x B300, Qwen3-8B, rank-16 all-linear adapters, input 1024 / output 256, csgmv backend",
             x=0.01, ha="left", fontsize=13, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig("multilora_summary.png", dpi=140, facecolor=SURF)
print("saved")
