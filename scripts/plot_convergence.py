"""Figure 6 (added 2026-10-04): after review, the final answer converges to G's.

One point per review direction: x = how often V answering alone differs from G in correctness,
y = how often the final answer differs from G in correctness (H06). Points below the diagonal converged.
Reads `rq2_cells.json` (no problem IDs) and writes `figures/fig6_convergence.{png,pdf}`, single column (3.3 in).

    python scripts/plot_convergence.py [--cells docs/analysis/claims/20261005_audit/rq2_cells.json]
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COLOR = {"stronger_reviews": "#6a4fc4", "weaker_reviews": "#e8711c"}  # as in the other figures
KIND_LABEL = {"stronger_reviews": "Stronger model reviews", "weaker_reviews": "Weaker model reviews"}
DOMAIN_MARKER = {"math": "o", "knowledge": "s", "code": "^", "nl_reasoning": "D"}
DOMAIN_LABEL = {"math": "Math", "knowledge": "Knowledge", "code": "Code", "nl_reasoning": "Reasoning"}
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "pdf.fonttype": 42,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7})


def domain_mean(rows, key):
    by = {}
    for r in rows:
        by.setdefault(r["domain"], []).append(r[key])
    return sum(sum(v) / len(v) for v in by.values()) / len(by)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=str(ROOT / "docs/analysis/claims/20261005_audit/rq2_cells.json"))
    ap.add_argument("--out", default=str(ROOT / "docs/paper_draft/figures/fig6_convergence"))
    args = ap.parse_args()
    rows = json.load(open(args.cells, encoding="utf-8"))["directions"]

    fig, ax = plt.subplots(figsize=(3.3, 3.0))
    top = 5 * (int(max(r["dis_before"] for r in rows)) // 5 + 2)
    ax.plot([0, top], [0, top], color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)), zorder=1)
    ax.text(top * 0.6, top * 0.6 + 1.2, "no change", rotation=45, fontsize=6.8, color=MUTED, ha="center")
    for r in rows:
        ax.scatter(r["dis_before"], r["dis_after"], s=14, marker=DOMAIN_MARKER[r["domain"]], color=COLOR[r["kind"]],
                   alpha=0.75, edgecolor=SURFACE, linewidth=0.4, zorder=3)
    for kind, color in COLOR.items():
        sub = [r for r in rows if r["kind"] == kind]
        x, y = domain_mean(sub, "dis_before"), domain_mean(sub, "dis_after")
        ax.scatter([x], [y], s=60, marker="X", color=color, edgecolor=INK, linewidth=0.8, zorder=4)
        ax.annotate(f"mean {x:.1f}% → {y:.1f}%", (x, y), xytext=(7, 0 if kind == "stronger_reviews" else -3),
                    textcoords="offset points", fontsize=6.8, color=INK, va="center")
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_aspect("equal")
    ax.set_xlabel("Differs from G, V answering alone (%)", fontsize=8, color=INK)
    ax.set_ylabel("Differs from G, final answer (%)", fontsize=8, color=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    handles = [Line2D([], [], marker="o", ls="", color=COLOR[k], markersize=5, label=KIND_LABEL[k]) for k in COLOR]
    handles += [Line2D([], [], marker=m, ls="", markerfacecolor="none", markeredgecolor=MUTED, markersize=4.5,
                       label=DOMAIN_LABEL[d]) for d, m in DOMAIN_MARKER.items()]
    ax.legend(handles=handles, loc="upper left", frameon=False, handletextpad=0.3, borderaxespad=0.2,
              labelspacing=0.3)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    print(f"{args.out}.png")


if __name__ == "__main__":
    main()
