"""RQ1 figures, drawn locally from committed claim reports (no problem IDs).

Figure 3 (`figures/fig3_outcome_levels`): for each review direction, which of five outcome levels the
final answer reaches, by who reviews whom and the capability gap. Reads H09_result.json.
Figure 4 (`figures/fig4_deltas_vs_gap`): Δ_G and Δ_S of every direction against the gap (H08_result.json),
and (c) the exact gap of every direction in each outcome level, without bins (H09_result.json).

    python scripts/plot_rq1.py [--claims docs/analysis/claims/20261005_audit --h08 docs/analysis/claims/20261005_audit/H08_result.json]

Sized for ACL: full text width 6.9 in, 8 pt text. Render and inspect the PNG for overlapping text.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
MAIN = "主分析：设计配对"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
KINDS = ["stronger_reviews", "weaker_reviews"]
KIND_COLOR = {"stronger_reviews": "#6a4fc4", "weaker_reviews": "#e8711c"}
KIND_LABEL = {"stronger_reviews": "Stronger model reviews", "weaker_reviews": "Weaker model reviews"}
# Five outcome levels: red for worse, gray for no change, one blue ramp for better.
LEVEL_COLOR = {1: "#d1453b", 2: "#d6d4cf", 3: "#79a5de", 4: "#2f6fc4", 5: "#123574"}
LEVEL_LABEL = {1: "Worse than G", 2: "About G", 3: "Above G, below stronger",
               4: "About the stronger", 5: "Beyond the stronger"}
LABEL_INK = {1: SURFACE, 2: INK, 3: INK, 4: SURFACE, 5: SURFACE}
DOMAIN_MARKER = {"math": "o", "knowledge": "s", "code": "^", "nl_reasoning": "D"}
DOMAIN_LABEL = {"math": "Math", "knowledge": "Knowledge", "code": "Code", "nl_reasoning": "Reasoning"}

plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "axes.labelcolor": INK,
                     "axes.titlesize": 8, "legend.fontsize": 7.5, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "pdf.fonttype": 42})


def style(ax, grid_axis="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def save(fig, out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix("." + ext), dpi=300, bbox_inches="tight", facecolor="white")
    print(out.with_suffix(".png"))


def figure1(h09, out):
    kinds = h09["analyses"][MAIN]["kinds"]
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.05), sharey=True)  # 2.55 until 2026-10-06; lowered to save page space
    for ax, kind in zip(axes, KINDS):
        bands = kinds[kind]["by_gap_band"]
        for x, b in enumerate(bands):
            bottom = 0.0
            for lv in range(1, 6):
                c = b["levels"][str(lv)]
                if not c:
                    continue
                h = 100 * c / b["n"]
                ax.bar(x, h, bottom=bottom, width=0.62, color=LEVEL_COLOR[lv], edgecolor="white",
                       linewidth=1.0, zorder=3)
                if h >= 6:
                    ax.text(x, bottom + h / 2, str(c), ha="center", va="center", fontsize=7.5 if h >= 9 else 6.5,
                            color=LABEL_INK[lv], zorder=4)
                bottom += h
        ax.set_xticks(range(len(bands)),
                      [f"{b['band'].replace('–', '–')}\n({b['n']} dir.)" for b in bands])
        ax.set_xlabel("Capability gap between the two models (pp)")
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_xlim(-0.55, len(bands) - 0.45)
        style(ax)
    axes[0].set_ylabel("Share of directions (%)")
    axes[0].set_title("(a) Stronger model reviews the weaker", loc="left", color=INK)
    axes[1].set_title("(b) Weaker model reviews the stronger (G is the stronger)", loc="left", color=INK)
    handles = [Patch(facecolor=LEVEL_COLOR[lv], edgecolor="none", label=LEVEL_LABEL[lv]) for lv in range(1, 6)]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.07),
               handlelength=1.2, columnspacing=1.2, handletextpad=0.4)
    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=1.5)
    save(fig, out)
    plt.close(fig)


def scatter_kind(ax, dirs, y):
    for d in dirs:
        lo, hi = d[y + "_ci95"]
        sig = lo > 0 or hi < 0
        col = KIND_COLOR[d["kind"]]
        ax.scatter(d["gap"], d[y], s=20, marker=DOMAIN_MARKER[d["domain"]], linewidth=0.8, zorder=3,
                   facecolor=col if sig else SURFACE, edgecolor=col, alpha=0.9)


LEVEL_TICK = {1: "<G", 2: "≈G", 3: "G–S", 4: "≈S", 5: ">S"}


def panel_levels(ax, h09, top):
    """Every direction's gap, by outcome level; side by side for the two review kinds. No bins."""
    rows = h09["analyses"][MAIN]["directions"]
    for lv in range(1, 6):
        for j, kind in enumerate(KINDS):
            vals = sorted(r["gap"] for r in rows if r["level"] == lv and r["kind"] == kind)
            x0 = lv + (j - 0.5) * 0.36
            for k, v in enumerate(vals):
                jitter = ((k * 7) % 5 - 2) * 0.035
                ax.scatter(x0 + jitter, v, s=11, color=KIND_COLOR[kind], edgecolor=SURFACE, linewidth=0.3,
                           alpha=0.9, zorder=3)
        n = [sum(1 for r in rows if r["level"] == lv and r["kind"] == k) for k in KINDS]
        ax.text(lv - 0.21, top + 1.2, str(n[0]), ha="center", va="bottom", fontsize=6.5, color=KIND_COLOR[KINDS[0]])
        ax.text(lv + 0.21, top + 1.2, str(n[1]), ha="center", va="bottom", fontsize=6.5, color=KIND_COLOR[KINDS[1]])
    ax.set_xticks(range(1, 6), [LEVEL_TICK[lv] for lv in range(1, 6)])
    ax.tick_params(axis="x", labelsize=7.5)
    ax.set_xlim(0.45, 5.55)
    ax.set_ylim(-1, top + 5)
    ax.set_ylabel("Capability gap (pp)")
    ax.set_xlabel("Final answer vs. G and S")
    ax.set_title("(c) Where each outcome occurs", loc="left", color=INK)
    style(ax)


def figure2(h08, h09, out):
    dirs = h08["analyses"][MAIN]["directions"]
    top = max(d["gap"] for d in dirs) + 2
    # 2.75 until 2026-10-06, 2.4 until 2026-10-10; legend moved to a right-hand column to save height
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 1.95), gridspec_kw={"width_ratios": [1, 1, 1.15]})
    ax = axes[0]
    ax.plot([0, top], [0, top], ls=(0, (3, 3)), color=MUTED, lw=0.9, zorder=1)
    ax.text(1, top - 2, "dashed: final = S", color=MUTED,
            fontsize=6.8, ha="left", va="top")
    scatter_kind(ax, dirs, "dG")
    ax.set_ylabel("Final − G alone (pp)")
    ax.set_title("(a) Δ_G: vs. G alone", loc="left", color=INK)
    ax = axes[1]
    scatter_kind(ax, dirs, "dC")
    ax.set_ylabel("Final − S alone (pp)")
    ax.set_title("(b) Δ_S: vs. the stronger alone", loc="left", color=INK)
    lo = min(min(d["dG"] for d in dirs), min(d["dC"] for d in dirs)) - 2
    axes[0].set_ylim(lo, top)
    axes[1].set_ylim(lo, max(d["dC"] for d in dirs) + 3)
    for a in axes[:2]:
        a.axhline(0, color=INK, lw=0.8, zorder=2)
        a.set_xlabel("Capability gap (pp)")
        a.set_xlim(-1, top)
        style(a)
    panel_levels(axes[2], h09, top)
    handles = [Line2D([], [], marker="o", ls="", markerfacecolor=KIND_COLOR[k], markeredgecolor=KIND_COLOR[k],
                      markersize=5, label=KIND_LABEL[k].replace(" reviews", "\nreviews")) for k in KINDS]
    handles += [Line2D([], [], marker=m, ls="", markerfacecolor=MUTED, markeredgecolor=MUTED, markersize=4.5,
                       label=DOMAIN_LABEL[dom]) for dom, m in DOMAIN_MARKER.items()]
    handles.append(Line2D([], [], marker="o", ls="", markerfacecolor=SURFACE, markeredgecolor=MUTED, markersize=5,
                          label="hollow: 95% CI\ncontains 0"))
    handles.append(Line2D([], [], ls="", label="S: stronger\nmodel alone"))
    fig.legend(handles=handles, loc="center left", ncol=1, frameon=False, bbox_to_anchor=(0.835, 0.5),
               handletextpad=0.3, labelspacing=0.5, fontsize=6.5)
    fig.tight_layout(rect=(0, 0, 0.84, 1), w_pad=0.8)
    save(fig, out)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims", default=str(ROOT / "docs/analysis/claims/20261005_audit"),
                    help="directory holding H09_result.json")
    ap.add_argument("--h08", default=str(ROOT / "docs/analysis/claims/20261005_audit/H08_result.json"))
    ap.add_argument("--figures", default=str(ROOT / "docs/paper_draft/figures"))
    args = ap.parse_args()
    h09 = json.load(open(Path(args.claims) / "H09_result.json", encoding="utf-8"))
    h08 = json.load(open(args.h08, encoding="utf-8"))
    figure1(h09, Path(args.figures) / "fig3_outcome_levels")
    figure2(h08, h09, Path(args.figures) / "fig4_deltas_vs_gap")


if __name__ == "__main__":
    main()
