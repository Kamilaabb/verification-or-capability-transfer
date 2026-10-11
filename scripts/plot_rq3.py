"""RQ3/RQ4 figure, drawn locally from the committed cost reports (no problem IDs).

Figure 7 (`figures/fig7_cost_accuracy`):
(a) every way of using the two models, as cost per question and accuracy relative to the stronger model alone:
    domain-equal means with their 95% intervals (H10);
(b) selector against each review direction, one point per pair: accuracy difference against cost ratio (H11).
Cost = parameters x (input + output tokens). JEV's API fee is not on the cost axis.

    python scripts/plot_rq3.py [--claims docs/analysis/claims/20261005_audit]

Sized for ACL: full text width 6.9 in, 8 pt text. Render and inspect the PNG for overlapping text.
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
REVIEW_COLOR = {"review_by_stronger": "#6a4fc4", "review_by_weaker": "#e8711c"}  # as in the other figures
SELECTOR = "#1a9a8a"
SYSTEM = {  # system: (label, color, marker, filled)
    "weaker_alone": ("Weaker model alone", "#9a9893", "o", True),
    "review_by_stronger": ("Stronger reviews weaker", REVIEW_COLOR["review_by_stronger"], "o", True),
    "review_by_weaker": ("Weaker reviews stronger", REVIEW_COLOR["review_by_weaker"], "o", True),
    "selector_primary": ("Selector", SELECTOR, "o", True),
    "JEV": ("JEV judge (+ API fee)", MUTED, "s", False),
    "oracle": ("Either model correct", MUTED, "D", False),
}
DOMAIN_MARKER = {"math": "o", "knowledge": "s", "code": "^", "nl_reasoning": "D"}

plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "axes.labelcolor": INK,
                     "axes.titlesize": 8, "legend.fontsize": 7, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "pdf.fonttype": 42})
REF = dict(color=MUTED, linewidth=0.7, linestyle=(0, (3, 3)), zorder=1)


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def save(fig, out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix("." + ext), dpi=300, bbox_inches="tight", facecolor="white")
    print(out.with_suffix(".png"))


LABEL_AT = {  # (label, offset in points, horizontal and vertical alignment)
    "weaker_alone": ("Weaker alone\n{y:+.1f} pp, {x:.1f}×", (7, 0), "left", "center"),
    "review_by_stronger": ("Stronger reviews\n{y:+.1f} pp, {x:.1f}×", (5, -5), "left", "top"),
    "review_by_weaker": ("Weaker reviews\n{y:+.1f} pp, {x:.1f}×", (-7, -4), "right", "top"),
    "selector_primary": ("Selector: {y:+.1f} pp, {x:.1f}×", (7, 0), "left", "center"),
    "JEV": ("JEV: {y:+.1f} pp, {x:.1f}× + fee", (7, 0), "left", "center"),
    "oracle": ("Either correct: {y:+.1f} pp, {x:.1f}×", (7, 0), "left", "center"),
}


def frontier(ax, h10):
    comps = {c["a"]: c for c in h10["comparisons"]}
    ax.axhline(0, **REF)
    ax.axvline(1, **REF)
    ax.scatter([1], [0], s=46, color=INK, zorder=5, edgecolor=SURFACE, linewidth=0.6)
    ax.annotate("Stronger alone", (1, 0), xytext=(4, -4), textcoords="offset points", fontsize=6.5,
                color=INK, ha="left", va="top")
    for name, (label, color, marker, filled) in SYSTEM.items():
        o = comps[name]["bootstrap"]["overall"]
        x, y = comps[name]["cost_ratio"]["main"]["overall"], o["mean_diff_pp"]
        ax.plot([x, x], o["ci95_pp"], color=color, linewidth=1.4, zorder=3, solid_capstyle="round")
        ax.scatter([x], [y], s=46, marker=marker, facecolor=color if filled else SURFACE, edgecolor=color,
                   linewidth=0.6 if filled else 1.3, zorder=4)
        text, offset, ha, va = LABEL_AT[name]
        ax.annotate(text.format(x=x, y=y), (x, y), xytext=offset, textcoords="offset points",
                    fontsize=6.5, color=INK, ha=ha, va=va)
    ax.set_xlim(0.75, 3.9)
    ax.set_ylim(-14.5, 10)
    ax.set_xlabel("Cost relative to stronger alone (×)")
    ax.set_ylabel("Accuracy vs. stronger alone (pp)")
    ax.set_title("(a) Every way of using the second model", loc="left", color=INK)
    style(ax)


def selector_vs_review(ax, h11):
    ax.axhline(0, **REF)
    ax.axvline(1, **REF)
    counts = {}
    for c in h11["comparisons"]:
        if c["a"] != "selector_primary":
            continue
        counts[c["b"]] = c["counts"]
        color = REVIEW_COLOR[c["b"]]
        for p in c["pairs"]:
            level = p["side"] == "level"
            ax.scatter(p["cost_ratio"]["main"], p["diff_pp"], s=15, marker=DOMAIN_MARKER[p["domain"]],
                       facecolor=SURFACE if level else color, edgecolor=color, linewidth=0.8, alpha=0.9, zorder=3)
    ax.set_xlim(0.35, 1.05)
    ax.set_xlabel("Selector cost / review cost (×)")
    ax.set_ylabel("Selector minus review (pp)")
    ax.set_title("(b) Selector against review, one point per pair", loc="left", color=INK)
    style(ax)
    return counts


def legends(fig, counts):
    names = {"math": "Math", "knowledge": "Knowledge", "code": "Code", "nl_reasoning": "Reasoning"}
    rows = [Line2D([], [], marker="o", ls="", color=REVIEW_COLOR[b], markersize=5,
                   label=f"Selector vs {SYSTEM[b][0].lower()}:" + chr(10) + f"{counts[b]['above']} pairs higher, "
                         f"{counts[b]['level']} level, {counts[b]['below']} lower")
            for b in ("review_by_stronger", "review_by_weaker")]
    marks = [Line2D([], [], marker="o", ls="", markerfacecolor=SURFACE, markeredgecolor=MUTED, markersize=5,
                    label="hollow: the pair's 95%" + chr(10) + "interval includes 0")]
    # domain shapes filled, so they are not read as "hollow = interval includes 0"
    marks += [Line2D([], [], marker=m, ls="", markerfacecolor=MUTED, markeredgecolor=MUTED, markersize=4.5,
                     label=names[d]) for d, m in DOMAIN_MARKER.items()]
    # one column on the right since 2026-10-10 (was two rows under the panels; space); legend text 6.3 -> 7.5
    # on 2026-10-12 (too small); panels keep their size, the figure grows wider and the paper includes it wider
    fig.legend(handles=rows + marks, loc="center left", bbox_to_anchor=(0.775, 0.5), ncol=1, frameon=False,
               handletextpad=0.3, labelspacing=0.6, fontsize=7.5)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--claims", type=Path, default=ROOT / "docs/analysis/claims/20261005_audit")
    ap.add_argument("--out", type=Path, default=ROOT / "docs/paper_draft/figures/fig7_cost_accuracy")
    args = ap.parse_args()
    h10 = json.loads((args.claims / "H10_result.json").read_text(encoding="utf-8"))
    h11 = json.loads((args.claims / "H11_result.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.5), gridspec_kw={"width_ratios": [1.2, 1]})  # 3.1 until 2026-10-10
    frontier(axes[0], h10)
    counts = selector_vs_review(axes[1], h11)
    fig.tight_layout(w_pad=1.5, rect=(0, 0, 0.78, 1))
    legends(fig, counts)
    save(fig, args.out)


if __name__ == "__main__":
    main()
