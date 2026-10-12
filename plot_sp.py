import json, os, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, INK2, MUTED, GRID, BASE, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SEQ = {8: "#86b6ef", 32: "#5598e7", 128: "#2a78d6", 256: "#1c5cab", 1024: "#0d366b"}
CS = [1, 8, 32, 128, 256]

sp = {}
for l in open("results_sp/results.jsonl"):
    d = json.loads(l); sp.setdefault(d["tag"], {})[d["conc"]] = d

def sp_series(tag, key):
    s = sp.get(tag, {}); cs = sorted(s)
    return cs, [s[c][key] for c in cs]

def lora_series(arm, key):
    by = {}
    for l in open(f"results/{arm}/bench.jsonl"):
        d = json.loads(l); by.setdefault(d["max_concurrency"], []).append(d[key])
    cs = sorted(by); return cs, [st.mean(by[c]) for c in cs]

def style(ax, title, ylabel):
    ax.set_facecolor(SURF); ax.set_xscale("log", base=2); ax.set_xticks(CS); ax.set_xticklabels([str(c) for c in CS])
    ax.minorticks_off(); ax.set_title(title, loc="left", fontsize=11.5, color=INK)
    ax.set_xlabel("Concurrent requests"); ax.set_ylabel(ylabel); ax.set_ylim(bottom=0)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

def line(ax, xs, ys, color, label, ls="-", marker="o"):
    ax.plot(xs, ys, ls, color=color, lw=2, marker=marker, ms=7, markeredgecolor=SURF, markeredgewidth=2, label=label, zorder=3)

plt.rcParams.update({"font.size": 10.5, "axes.edgecolor": BASE, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
fig, ax = plt.subplots(2, 2, figsize=(15, 10.5), facecolor=SURF)

# Row 1: ideal multi-SP (per-tenant 1024-token prefix, prefix cache ON), by tenant count
for a, key, title, yl in [(ax[0, 0], "itl_p50", "A. Soft prompt with prefix cache: per-token latency", "ITL p50 (ms)"),
                          (ax[0, 1], "out_tput", "B. Soft prompt with prefix cache: output throughput", "tok/s")]:
    line(a, *sp_series("A_nosp", key), MUTED, "No soft prompt", "--")
    for n, c in SEQ.items():
        line(a, *sp_series(f"C_sp_N{n}", key), c, f"{n} tenants x 1024 soft tokens")
    style(a, title, yl)
ax[0, 0].legend(frameon=False, loc="upper left", labelcolor=INK2)

# Row 2: SP vs LoRA at 256 tenants
cmp = [("nosp", "No soft prompt / no LoRA", MUTED, "--", "o"),
       ("spc", "Soft prompt, 256 tenants, prefix cache (ideal framework)", "#2a78d6", "-", "o"),
       ("spn", "Soft prompt, no prefix cache (SGLang today)", "#eb6834", "-", "s"),
       ("lora", "LoRA r16, 256 adapters in GPU", "#1baf7a", "-", "^")]
for a, spkey, lkey, title, yl, scale in [
        (ax[1, 0], "out_tput", "output_throughput", "C. 256 tenants: soft prompt vs LoRA, throughput", "tok/s", 1),
        (ax[1, 1], "ttft_p50", "median_ttft_ms", "D. 256 tenants: time to first token", "TTFT p50 (ms)", 1)]:
    for k, label, color, ls, mk in cmp:
        if k == "lora":
            xs, ys = lora_series("256", lkey)
        else:
            xs, ys = sp_series({"nosp": "A_nosp", "spc": "C_sp_N256", "spn": "B_sp_nocache"}[k], spkey)
        ys = [y * scale for y in ys]
        line(a, xs, ys, color, label, ls, mk)
        short = {"nosp": "Baseline", "spc": "SP + cache", "spn": "SP, no cache", "lora": "LoRA r16"}[k]
        dy = {"nosp": 0, "spc": -9 if spkey == "ttft_p50" else 0, "spn": 0, "lora": 0}[k]
        if spkey == "ttft_p50" and k == "nosp":
            dy = 9
        if xs:
            a.annotate(short, (xs[-1], ys[-1]), xytext=(6, dy), textcoords="offset points",
                       va="center", fontsize=9, color=INK2)
    style(a, title, yl); a.set_xlim(right=900)
ax[1, 0].legend(frameon=False, loc="upper left", labelcolor=INK2)

fig.suptitle("Soft prompt (1024 tokens / tenant) vs LoRA (rank 16) multi-tenant serving - SGLang 0.5.21, Qwen3-8B, 1x B300, input 1024 / output 256",
             x=0.01, ha="left", fontsize=12.5, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig("sp_vs_lora.png", dpi=140, facecolor=SURF)
print("saved")
