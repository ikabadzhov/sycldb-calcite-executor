#!/usr/bin/env python3
"""
Plot SSB benchmark results: warm average execution time per query,
with a stacked breakdown of Kernel vs Transfer (load) vs Other time.
"""

import csv
import sys
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

CSV_PATH = "/tmp/ssb_optimized.csv"
OUT_PATH = "/tmp/ssb_optimized_results.png"

# ── load CSV ──────────────────────────────────────────────────────────────────
rows = defaultdict(list)          # query -> list of (total, jit, kernel, transfer, parse, other)
with open(CSV_PATH) as f:
    reader = csv.DictReader(f)
    for r in reader:
        query = r["Query"].replace(".sql", "").upper()
        rep   = int(r["Repetition"])
        if rep == 1:
            continue           # skip cold run
        rows[query].append({
            "total":    float(r["TotalTime_ms"]),
            "jit":      float(r["JIT_ms"]),
            "kernel":   float(r["Kernel_ms"]),
            "transfer": float(r["Transfer_ms"]),
            "parse":    float(r["Parse_ms"]),
            "other":    float(r["Other_ms"]),
        })

SSB_ORDER = [
    "Q11","Q12","Q13",
    "Q21","Q22","Q23",
    "Q31","Q32","Q33","Q34",
    "Q41","Q42","Q43",
]

def avg(lst, key):
    vals = [d[key] for d in lst]
    return sum(vals) / len(vals) if vals else 0.0

queries  = [q for q in SSB_ORDER if q in rows]
totals   = [avg(rows[q], "total")    for q in queries]
kernels  = [avg(rows[q], "kernel")   for q in queries]
transfers= [avg(rows[q], "transfer") for q in queries]
others   = [avg(rows[q], "other")    for q in queries]

# ── plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 7))
fig.patch.set_facecolor("#0d1117")
ax.set_facecolor("#161b22")

x  = np.arange(len(queries))
bw = 0.55

colors = {
    "kernel":   "#58a6ff",
    "transfer": "#3fb950",
    "other":    "#e3b341",
}

b_kernel   = ax.bar(x, kernels,   bw, label="GPU Kernel",      color=colors["kernel"],   zorder=3)
b_transfer = ax.bar(x, transfers, bw, label="Data Transfer",   color=colors["transfer"], bottom=kernels, zorder=3)
b_other    = ax.bar(x, others,    bw, label="Other (CPU/misc)",color=colors["other"],
                    bottom=[k+t for k,t in zip(kernels, transfers)], zorder=3)

# value labels on top of each bar
for i, tot in enumerate(totals):
    ax.text(i, tot + max(totals)*0.01, f"{tot:.0f}", ha="center", va="bottom",
            color="white", fontsize=7.5, fontweight="bold")

# grid & axes
ax.set_xticks(x)
ax.set_xticklabels(queries, color="white", fontsize=10)
ax.set_ylabel("Execution Time (ms) — warm avg of reps 2–5", color="#8b949e", fontsize=10)
ax.set_xlabel("SSB Query", color="#8b949e", fontsize=10)
ax.set_title("SYCLDB JIT-Fused Engine — SSB SF100 on NVIDIA L40S\n"
             "(sycl::reduction · const-ref ctx · branchless literals)",
             color="white", fontsize=13, fontweight="bold", pad=16)

ax.tick_params(colors="#8b949e")
ax.spines[:].set_color("#30363d")
ax.yaxis.set_tick_params(labelcolor="#8b949e")
ax.grid(axis="y", color="#21262d", linewidth=0.8, zorder=0)
ax.set_xlim(-0.5, len(queries) - 0.5)
ax.set_ylim(0, max(totals) * 1.13)

legend = ax.legend(
    handles=[
        mpatches.Patch(color=colors["kernel"],   label="GPU Kernel"),
        mpatches.Patch(color=colors["transfer"],  label="Data Transfer"),
        mpatches.Patch(color=colors["other"],     label="Other (CPU/misc)"),
    ],
    loc="upper right", facecolor="#161b22", edgecolor="#30363d",
    labelcolor="white", fontsize=9
)

# group separators
for sep in [2.5, 5.5, 9.5]:
    ax.axvline(sep, color="#30363d", linewidth=1.2, linestyle="--", zorder=2)

# group labels
group_info = [(1,"Q1.x"),(4,"Q2.x"),(7.5,"Q3.x"),(11,"Q4.x")]
for gx, gl in group_info:
    ax.text(gx, max(totals)*1.07, gl, ha="center", color="#8b949e",
            fontsize=8.5, fontstyle="italic")

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved → {OUT_PATH}")
