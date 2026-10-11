"""RQ2 figure (Figure 5, one row of four panels; 2x2 until 2026-10-06, changed to save page space): why review misses what a second model could add.

(a) relative to the stronger model alone: the best of the two solo answers per question (oracle), the best of
    G's solo answer and the final answer, and the final answer itself;
(b) on questions where exactly one model is right alone, how often the final answer is right;
(c) the gain over G alone split by which model was right alone;
(d) weaker model reviews the stronger: how often the final answer stays right where only G was right
    alone, against Δ_S, coloured by the H09 outcome level.

Reads `rq2_cells.json` (written by scripts/rq2_cells.py from the claim reports; no problem IDs) and
H09_result.json, and writes `figures/fig5_mechanism.{png,pdf}` next to the paper draft. Drawn locally.

    python scripts/plot_rq2.py [--cells docs/analysis/claims/20261005_audit/rq2_cells.json --h09 docs/analysis/claims/20261005_audit/H09_result.json]
"""
import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
COLOR = {"stronger_reviews": "#6a4fc4", "weaker_reviews": "#e8711c"}
KIND_LABEL = {"stronger_reviews": "Stronger model reviews", "weaker_reviews": "Weaker model reviews"}
LEVEL_COLOR = {1: "#d1453b", 2: "#d6d4cf", 5: "#123574"}
LEVEL_LABEL = {1: "worse", 2: "about", 5: "beyond"}
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "pdf.fonttype": 42})
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
KINDS = ["stronger_reviews", "weaker_reviews"]
REF = dict(color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)), zorder=1)


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=6.8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def label(ax, x, y):
    ax.set_xlabel(x, fontsize=8, color=INK)
    ax.set_ylabel(y, fontsize=8, color=INK)


def dots(ax, rows, x, y):
    for r in rows:
        ax.scatter(r[x], r[y], s=11, color=COLOR[r["kind"]], alpha=0.75, edgecolor=SURFACE, linewidth=0.4, zorder=3)


def strips(ax, rows, summary, groups, ylim, fmt):
    rnd = random.Random(0)
    for gi, (key, _) in enumerate(groups):
        for ki, kind in enumerate(KINDS):
            x0 = gi + (ki - 0.5) * 0.38
            vals = [r[key] for r in rows if r["kind"] == kind and r[key] is not None]
            ax.scatter([x0 + rnd.uniform(-0.09, 0.09) for _ in vals], vals, s=8, color=COLOR[kind],
                       alpha=0.55, edgecolor="none", zorder=3)
            m = summary[kind][key]["mean"]
            ax.plot([x0 - 0.15, x0 + 0.15], [m, m], color=INK, linewidth=1.4, zorder=4)
            ax.text(x0, ylim[1] - (0.07 + 0.075 * ki) * (ylim[1] - ylim[0]), fmt.format(m), ha="center", va="bottom",
                    fontsize=6.5, color=COLOR[kind])
    ax.set_xticks(range(len(groups)), [g for _, g in groups])
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylim(*ylim)


def panel_headroom(ax, rows, summary):
    groups = [("oracle_before", "G or V"), ("oracle_after", "G or\nfinal"),
              ("dC", "Final")]
    strips(ax, rows, summary, groups, (-14, 23), "{:+.1f}")
    ax.axhline(0, color=MUTED, linewidth=0.6)
    label(ax, "Right answer from", "vs. stronger alone (pp)")
    ax.set_title("(a) The room shrinks\nafter review", fontsize=8, color=INK, loc="left")


def panel_disagreement(ax, rows, summary):
    groups = [("keep_rate", "Only G\nright"), ("adopt_rate", "Only V\nright")]
    strips(ax, rows, summary, groups, (0, 128), "{:.0f}%")
    ax.set_yticks([0, 25, 50, 75, 100])
    label(ax, "Only one model right alone", "Final answer right (%)")
    ax.set_title("(b) The final answer\nfollows G", fontsize=8, color=INK, loc="left")


def panel_keep(ax, rows, levels):
    for r in rows:
        if r["kind"] != "weaker_reviews":
            continue
        lv = levels[r["id"]]
        ax.scatter(r["keep_rate"], r["dC"], s=18, color=LEVEL_COLOR[lv], zorder=3,
                   edgecolor=MUTED if lv == 2 else SURFACE, linewidth=0.5)
    ax.axhline(0, color=MUTED, linewidth=0.6)
    label(ax, "Only G right alone:\nfinal answer still right (%)", "Δ_S (pp)")
    ax.set_title("(d) Weaker model reviews:\nkeeping G's answer", fontsize=8, color=INK, loc="left")
    handles = [Line2D([], [], marker="o", ls="", markerfacecolor=LEVEL_COLOR[lv],
                      markeredgecolor=MUTED if lv == 2 else LEVEL_COLOR[lv], markersize=5, label=LEVEL_LABEL[lv])
               for lv in (5, 2, 1)]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=6.5, handletextpad=0.1, borderaxespad=0.1,
              title="vs. the stronger", title_fontsize=6.5, alignment="left", labelspacing=0.25)
    ax.set_xlim(38, 102)
    ax.set_ylim(-13, 9)
    ax.set_yticks([-10, -5, 0, 5])


def panel_split(ax, summary):
    parts = [("gain_from_v", "Only\nV"), ("loss_from_g", "Only\nG"),
             ("net_both", "Both/\nnone"), ("total", "Total\n ")]
    width = 0.36
    for ki, kind in enumerate(KINDS):
        s = summary[kind]
        vals = [s["gain_from_v"]["mean"], -s["loss_from_g"]["mean"], s["net_both"]["mean"]]
        vals.append(sum(vals))
        for pi, v in enumerate(vals):
            x = pi + (ki - 0.5) * width
            ax.bar(x, v, width=width - 0.04, color=COLOR[kind], edgecolor=SURFACE, linewidth=0.6, zorder=3)
            ax.text(x + (ki - 0.5) * 0.14, v + (0.25 if v >= 0 else -0.25), f"{v:+.1f}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=6, color=INK)
    ax.axhline(0, color=MUTED, linewidth=0.6)
    ax.axvline(2.5, color=GRID, linewidth=0.8)
    ax.set_xticks(range(len(parts)), [p for _, p in parts])
    ax.set_ylim(-4, 10)
    label(ax, "Which model is right alone", "Gain over G (pp)")
    ax.set_title("(c) Where the gain\ncomes from", fontsize=8, color=INK, loc="left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=str(ROOT / "docs/analysis/claims/20261005_audit/rq2_cells.json"))
    ap.add_argument("--h09", default=str(ROOT / "docs/analysis/claims/20261005_audit/H09_result.json"))
    ap.add_argument("--out", default=str(ROOT / "docs/paper_draft/figures/fig5_mechanism"))
    args = ap.parse_args()
    data = json.load(open(args.cells, encoding="utf-8"))
    rows, summary = data["directions"], data["summary"]
    levels = {d["id"]: d["level"] for d in
              json.load(open(args.h09, encoding="utf-8"))["analyses"]["主分析：设计配对"]["directions"]}

    # 2.65 with a legend row on top until 2026-10-10; colours as in Figure 4, explained in the caption (space)
    fig, axes = plt.subplots(1, 4, figsize=(6.9, 2.3), layout="constrained",
                             gridspec_kw={"width_ratios": [1.15, 0.95, 1.1, 1.05]})
    fig.get_layout_engine().set(w_pad=0.04, wspace=0.06)
    for ax in axes:
        style(ax)
    panel_headroom(axes[0], rows, summary)
    panel_disagreement(axes[1], rows, summary)
    panel_split(axes[2], summary)
    panel_keep(axes[3], rows, levels)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    print(f"{args.out}.png")


if __name__ == "__main__":
    main()
