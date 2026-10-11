"""Figure 1, introduction overview (`figures/fig1_overview`), drawn locally from committed reports (no problem IDs).

(a, top row) the setup: G answers; V answers alone and, separately, reviews G's answer; a selector picks
    between the two solo answers. (b)-(d), bottom row, one small plot per main finding:
(b) review often lifts G but rarely beats the stronger model alone (H09 outcome levels, design pairs);
(c) after reading G's answer, the reviewer mostly ends where G is (H06 / rq2_cells);
(d) with the same two calls, picking one of two solo answers beats review (H10).

    python scripts/plot_overview.py

Full text width 6.9 in, 8 pt text. Render and inspect the PNG for overlapping text.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
KIND_COLOR = {"stronger_reviews": "#6a4fc4", "weaker_reviews": "#e8711c"}  # as in the other figures
SELECTOR, SOLO = "#1a9a8a", "#9a9893"
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "pdf.fonttype": 42,
                     "xtick.labelsize": 7, "ytick.labelsize": 7})
MAIN = "主分析：设计配对"


def style(ax, grid="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, length=2)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def box(ax, x, y, text, color=INK):
    return ax.text(x, y, text, ha="center", va="center", fontsize=7, color=INK, zorder=3,
                   bbox=dict(boxstyle="round,pad=0.35", facecolor=SURFACE, edgecolor=color, linewidth=0.9))


def extent(ax, artist):
    """Data-coordinate extent of a text artist, including its box."""
    ax.figure.canvas.draw()
    patch = artist.get_bbox_patch()
    bb = (patch or artist).get_window_extent().transformed(ax.transData.inverted())
    return bb


def arrow(ax, a, b):
    ax.annotate("", xy=b, xytext=a, zorder=2,
                arrowprops=dict(arrowstyle="-|>", color=MUTED, linewidth=0.9, mutation_scale=7, shrinkA=1, shrinkB=1))


YMAX = 26
REVIEW_BOX = "#b8860b"  # dark gold edge and title, no fill
G_COLOR, V_COLOR = "#3f6fb5", "#4a8c5c"   # the two models in panel (a); not the stronger/weaker colours
RIGHT, WRONG = "#2e8b3e", "#c0392b"


def robot(ax, x, y, color, label, s=1.0):
    """A small robot head centred at (x, y) in an equal-aspect axes; s scales it."""
    from matplotlib.patches import Circle, FancyBboxPatch
    w, h = 4.4 * s, 3.4 * s
    ax.plot([x, x], [y + h / 2, y + h / 2 + 0.9 * s], color=color, linewidth=1.1, zorder=3)
    ax.add_patch(Circle((x, y + h / 2 + 1.05 * s), 0.38 * s, facecolor=color, edgecolor="none", zorder=3))
    for side in (-1, 1):  # ears
        ax.add_patch(FancyBboxPatch((x + side * w / 2 - (0.55 * s if side > 0 else 0), y - 0.6 * s), 0.55 * s,
                                    1.2 * s, boxstyle="round,pad=0,rounding_size=0.2", facecolor=color,
                                    edgecolor="none", zorder=2))
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle=f"round,pad=0,rounding_size={0.9 * s}",
                                facecolor=color, edgecolor="none", zorder=3))
    ax.add_patch(FancyBboxPatch((x - w / 2 + 0.55 * s, y - h / 2 + 0.55 * s), w - 1.1 * s, h - 1.1 * s,
                                boxstyle=f"round,pad=0,rounding_size={0.6 * s}", facecolor="white",
                                edgecolor="none", zorder=4))
    for side in (-1, 1):  # eyes
        ax.add_patch(Circle((x + side * 0.75 * s, y + 0.15 * s), 0.32 * s, facecolor=color, edgecolor="none",
                            zorder=5))
    ax.plot([x - 0.55 * s, x + 0.55 * s], [y - 0.65 * s, y - 0.65 * s], color=color, linewidth=1.0,
            solid_capstyle="round", zorder=5)
    ax.text(x, y - h / 2 - 0.5, label, ha="center", va="top", fontsize=7, color=color, fontweight="bold")


def card(ax, x, y, answer, ok, caption=None):
    """An answer card: the answer and a tick or cross; ok=None draws no mark."""
    mark = "" if ok is None else (" ✓" if ok else " ✗")
    t = ax.text(x, y, answer, ha="center", va="center", fontsize=7.5, color=INK, zorder=6,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=MUTED, linewidth=0.7))
    if mark:
        ax.annotate(mark, xy=(1, 0.5), xycoords=t, ha="left", va="center", fontsize=8, fontweight="bold",
                    color=RIGHT if ok else WRONG, zorder=6)
    if caption:
        ax.text(x, y - 2.4, caption, ha="center", va="top", fontsize=6.3, color=MUTED)
    return t


def region(ax, x0, y0, x1, y1, color, title, note, fill=None):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=1.2",
                                facecolor=fill or color + "14", edgecolor=color, linewidth=0.9, zorder=1))
    ax.text(x0 + 1.0, y1 - 0.9, title, ha="left", va="top", fontsize=7.8, color=color)  # not bold: bold DejaVu kerned "Revie w" in some PDF viewers
    ax.text(x1 - 1.0, y1 - 0.9, note, ha="right", va="top", fontsize=6.3, color=MUTED, style="italic")


def flow(ax, a, b, color=MUTED, rad=0.0):
    ax.annotate("", xy=b, xytext=a, zorder=2,
                arrowprops=dict(arrowstyle="-|>", color=color, linewidth=0.9, mutation_scale=7, shrinkA=0,
                                shrinkB=0, connectionstyle=f"arc3,rad={rad}"))


def setup(ax):
    """Panel (a), drawn with the real example of Figure 2 (Llama-3.1-8B as G, Qwen3-8B as V; answer 11).

    Data units are square: main() sizes the axes so that 100 x-units and YMAX y-units have the same length.
    """
    ax.set_xlim(0, 100)
    ax.set_ylim(0, YMAX)
    ax.axis("off")
    review_c, select_c = REVIEW_BOX, SELECTOR  # review box neutral: purple means "stronger reviews" in (b)-(d)
    ax.text(5.5, 13, "Question", ha="center", va="center", fontsize=7.5, color=INK, zorder=3,
            bbox=dict(boxstyle="round,pad=0.5", facecolor=SURFACE, edgecolor=INK, linewidth=0.9))
    # both models answer alone
    robot(ax, 18.5, 19.5, G_COLOR, "G")
    robot(ax, 18.5, 6.5, V_COLOR, "V")
    flow(ax, (10.6, 14.2), (15.4, 18.6))
    flow(ax, (10.6, 11.8), (15.4, 7.4))
    card(ax, 27.5, 19.5, "10", False, caption="G alone")
    card(ax, 27.5, 6.5, "11", True, caption="V alone")
    flow(ax, (21.7, 19.5), (25.4, 19.5))
    flow(ax, (21.7, 6.5), (25.4, 6.5))
    # review: V reads G's answer and gives the final answer
    region(ax, 36, 14.2, 67, 25.6, review_c, "Review", "2 model calls", fill="none")
    flow(ax, (32.2, 19.5), (35.6, 19.3), color=G_COLOR)  # arrows stop at the box edge
    ax.text(37.2, 20.4, "G's answer", ha="left", va="bottom", fontsize=6.3, color=G_COLOR, style="italic")
    robot(ax, 47, 18.6, V_COLOR, "V", s=0.85)
    card(ax, 59.5, 18.6, "10", False, caption="final answer")
    flow(ax, (50.0, 18.6), (56.6, 18.6))
    # selection: pick one of the two solo answers
    region(ax, 36, 0.4, 67, 11.8, select_c, "Select", "2 model calls")
    ax.text(46, 5.6, "selector", ha="center", va="center", fontsize=7, color=INK, zorder=3,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor=select_c, linewidth=0.9))
    flow(ax, (31.4, 18.2), (35.6, 9.6), color=G_COLOR, rad=-0.1)
    flow(ax, (31.4, 6.5), (35.6, 6.3), color=V_COLOR)
    # The selector's pick for this question (2026-10-10): primary math selector, out-of-fold on the development set
    # (runs/20261002T_math_selector_oof_2800f9a/math__c207__c201, pick_v on all five seeds, p_V 0.89 vs p_G 0.33).
    card(ax, 59.5, 5.6, "11", True, caption="picked answer")
    flow(ax, (51.4, 5.6), (54.6, 5.6))
    # the two reference points
    ax.text(70, 21.5, "Compare the final answer with", ha="left", va="center", fontsize=7, color=INK)
    # plain "Δ_G" as in Figures 4 and 5 (mathtext made the subscript too small)
    ax.text(71, 17.8, "Δ_G:  G alone (the usual measure)", ha="left", va="center", fontsize=7, color=INK)
    ax.text(71, 14.4, "Δ_S:  the stronger model alone", ha="left", va="center", fontsize=7, color=INK)
    ax.text(70, 6.5, "In this example V is the stronger\nmodel. Alone it answers 11, but\nafter reading G's 10 it answers 10.",
            ha="left", va="center", fontsize=6.5, color=MUTED, linespacing=1.35)
    ax.text(0.0, 1.03, "(a) Review and selection use the same two models; we compare both with each model alone",
            fontsize=8, color=INK, transform=ax.transAxes)


def finding_levels(ax, h09):
    """Three exclusive outcomes (2026-10-10): the old two groups counted the eight weaker-reviews directions
    twice, because when the weaker model reviews, G is the stronger model and "better than G" is "better than the
    stronger". Directions no different from G (level 2) are not drawn."""
    c = Counter((d["kind"], d["level"]) for d in h09["analyses"][MAIN]["directions"])
    n = {k: sum(v for (kk, _), v in c.items() if kk == k) for k in KIND_COLOR}
    groups = (("Worse\nthan G", {k: c[(k, 1)] for k in KIND_COLOR}),
              ("Better than\nG only", {k: c[(k, 3)] + c[(k, 4)] for k in KIND_COLOR}),
              ("Better than\nstronger", {k: c[(k, 5)] for k in KIND_COLOR}))
    for i, (name, vals) in enumerate(groups):
        for j, k in enumerate(KIND_COLOR):
            v = vals[k]
            x = i + (j - 0.5) * 0.4
            if k == "weaker_reviews" and i == 1:  # impossible: G is the stronger model
                ax.text(x, 1, "n/a", ha="center", va="bottom", fontsize=6, color=MUTED)
                continue
            ax.bar(x, v, width=0.38, color=KIND_COLOR[k], zorder=3)
            if v > 30:  # tall bar: label inside, clear of the legend
                ax.text(x, v - 1.5, f"{v}", ha="center", va="top", fontsize=6.5, color="white")
            else:
                ax.text(x, v + 1, f"{v}", ha="center", va="bottom", fontsize=6.5, color=INK)
    ax.set_xticks(range(3), [g[0] for g in groups], fontsize=6.2)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=KIND_COLOR[k]) for k in KIND_COLOR],
              labels=["Stronger\nreviews", "Weaker\nreviews"], loc="upper right", frameon=False, fontsize=6.2,
              handlelength=1, handletextpad=0.4)
    ax.set_ylim(0, 66)  # legend sits above the 8 bar
    ax.set_yticks([0, 20, 40])
    ax.set_ylabel(f"Review directions (of {n['stronger_reviews']})", fontsize=7, color=INK)
    ax.set_title("(b) Review lifts G,\nrarely past the stronger", loc="left", fontsize=8, color=INK)
    style(ax)


def finding_convergence(ax, cells):
    rows = cells["directions"]

    def mean(rs, key):
        by = {}
        for r in rs:
            by.setdefault(r["domain"], []).append(r[key])
        return sum(sum(v) / len(v) for v in by.values()) / len(by)

    bars = [("V alone", mean(rows, "dis_before"), SOLO)]
    for k, label in (("stronger_reviews", "Stronger\nreviews"), ("weaker_reviews", "Weaker\nreviews")):
        bars.append((label, mean([r for r in rows if r["kind"] == k], "dis_after"), KIND_COLOR[k]))
    for i, (label, v, color) in enumerate(bars):
        ax.bar(i, v, width=0.6, color=color, zorder=3)
        ax.text(i, v + 0.6, f"{v:.1f}%", ha="center", va="bottom", fontsize=6.5, color=INK)
    ax.set_xticks(range(3), [b[0] for b in bars])
    ax.set_ylim(0, 30)
    ax.set_ylabel("Differs from G (%)", fontsize=7, color=INK)
    ax.set_title("(c) After reading G,\nV ends where G is", loc="left", fontsize=8, color=INK)
    style(ax)


def finding_cost(ax, h10):
    comps = {c["a"]: c for c in h10["comparisons"]}
    items = [("Stronger\nreviews", "review_by_stronger", KIND_COLOR["stronger_reviews"]),
             ("Weaker\nreviews", "review_by_weaker", KIND_COLOR["weaker_reviews"]),
             ("Selector", "selector_primary", SELECTOR),
             ("Oracle\nupper", "oracle", SOLO)]  # 2026-10-10: how far everything is from the oracle
    for i, (label, key, color) in enumerate(items):
        o = comps[key]["bootstrap"]["overall"]
        v, (lo, hi) = o["mean_diff_pp"], o["ci95_pp"]
        if key == "oracle":  # an upper bound, not a method: outlined, no cost
            ax.bar(i, v, width=0.6, facecolor="white", edgecolor=MUTED, hatch="////", linewidth=0.8, zorder=3)
        else:
            ax.bar(i, v, width=0.6, color=color, zorder=3)
        ax.plot([i, i], [lo, hi], color=INK, linewidth=0.9, zorder=4)
        ax.text(i, (hi + 0.3) if v > 0 else (lo - 0.3), f"{v:+.1f}", ha="center",
                va="bottom" if v > 0 else "top", fontsize=6.5, color=INK)
    ax.axhline(0, color=MUTED, linewidth=0.7)
    ticks = [(label if "\n" in label else label + "\n ")
             + ("\nbound" if key == "oracle" else f"\n{comps[key]['cost_ratio']['main']['overall']:.1f}× cost")
             for label, key, _ in items]
    ax.set_xticks(range(len(items)), ticks, fontsize=5.8)
    ax.set_ylim(-5.5, 10)
    ax.set_yticks(range(-4, 9, 4))
    ax.set_ylabel("vs. stronger alone (pp)", fontsize=7, color=INK)
    ax.set_title("(d) Same two calls:\npicking beats review", loc="left", fontsize=8, color=INK)
    style(ax)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h09", type=Path, default=ROOT / "docs/analysis/claims/20261005_audit/H09_result.json")
    ap.add_argument("--cells", type=Path, default=ROOT / "docs/analysis/claims/20261005_audit/rq2_cells.json")
    ap.add_argument("--h10", type=Path, default=ROOT / "docs/analysis/claims/20261005_audit/H10_result.json")
    ap.add_argument("--out", type=Path, default=ROOT / "docs/paper_draft/figures/fig1_overview")
    args = ap.parse_args()
    load = lambda p: json.loads(p.read_text(encoding="utf-8"))
    W, H = 6.9, 3.5  # 4.0 until 2026-10-06; less space between (a) and (b)-(d), shorter (b)-(d); (a) kept full size
    fig = plt.figure(figsize=(W, H))
    top_w = 0.98 * W  # panel (a): square data units, 100 wide and YMAX tall
    top_h = top_w * YMAX / 100
    setup(fig.add_axes([0.01, 1 - (top_h + 0.25) / H, 0.98, top_h / H]))
    gs = fig.add_gridspec(1, 3, left=0.08, right=0.99, bottom=0.12, top=1 - (top_h + 0.6) / H, wspace=0.5,
                      width_ratios=[1.1, 0.85, 1.25])  # (d) has four bars since 2026-10-10
    finding_levels(fig.add_subplot(gs[0, 0]), load(args.h09))
    finding_convergence(fig.add_subplot(gs[0, 1]), load(args.cells))
    finding_cost(fig.add_subplot(gs[0, 2]), load(args.h10))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(args.out.with_suffix("." + ext), dpi=300, bbox_inches="tight", facecolor="white")
    print(args.out.with_suffix(".png"))


if __name__ == "__main__":
    main()
