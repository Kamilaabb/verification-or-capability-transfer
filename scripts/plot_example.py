"""Figure 2 (`figures/fig2_example`): one real review in which the reviewer drops its own right answer.

Candidate A07 of `docs/analysis/example_candidates_20261004/candidates.md` (math development set; Llama-3.1-8B
answers, Qwen3-8B answers alone and reviews). Picked on 2026-10-04 from the ranked candidates
because the gold answer is unambiguous and the reviewer's own text shows the copying: it states the right bound
("less than or equal to 31") and then copies G's list, which stops at 29. The excerpts are verbatim (Markdown
bold and LaTeX delimiters removed, cuts marked [...]); `check_verbatim` asserts that against the candidates file.

    python scripts/plot_example.py

Single column, 3.3 in wide, 7 pt text. Render and inspect the PNG for overflow.
"""
import argparse
import re
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
SOLO, STRONGER_REVIEWS = "#9a9893", "#6a4fc4"  # as in the other figures
RIGHT, WRONG = "#1a7f37", "#b42318"
MARK = "#fde68a"  # highlighter behind the line the caption points to
plt.rcParams.update({"font.size": 7, "font.family": "DejaVu Sans", "pdf.fonttype": 42})

QUESTION = "How many natural numbers less than 1000 have exactly three distinct positive integer divisors?"
GOLD = "11"
# 2026-10-10: one-line solution so readers can check the math, which model is stronger, and what V was told.
SOLUTION = "these are the squares of primes; p² < 1000 gives p ≤ 31, and 11 primes are ≤ 31."
# Math review prompt, lines 1 and 3 of its instruction (appendix, Review prompts).
TOLD = ['Told: "The proposed solution may be correct or incorrect. [...]',
        'If the proposed solution is correct, retain it. If it contains any',
        'mathematical error, correct the error [...]"']
# (title, colour, lines, answer, right?). A line is (text, highlighted?).
PANELS = [
    ("Llama-3.1-8B answers (G, weaker)", SOLO, [
        ("[...] we need to find the prime numbers less than 31.", False),
        ("The prime numbers less than 31 are 2, 3, 5, 7, 11, 13,", True),
        ("17, 19, 23, and 29. [...] There are 10 such numbers.", True),
    ], "10", False),
    ("Qwen3-8B alone (V, stronger)", SOLO, [
        ("[...] 17, 19, 23, 29, 31 [...] 31^2 = 961 [...]", True),
        ("All of these are less than 1000.", False),
        ("So, there are 11 such numbers.", False),
    ], "11", True),
    ("Qwen3-8B reviews G's answer (V)", STRONGER_REVIEWS, [
        ("[...] we need to find all prime numbers less than 31.62,", False),
        ("i.e., less than or equal to 31.", True),
        ("The prime numbers less than or equal to 31 are:", False),
        ("2, 3, 5, 7, 11, 13, 17, 19, 23, 29", True),
        ("That's a total of 10 primes.", False),
    ], "10", False),
]


def plain(s):
    s = re.sub(r"\*\*|\$\$?|\\boxed|\\\\", "", s)
    s = s.replace("&=", "=").replace("’", "'").replace("\\approx", "≈").replace("\\Rightarrow", "⇒")
    return re.sub(r"\s+", " ", s)


def check_verbatim(path):
    """Every excerpt, cut at its [...] marks, must occur in candidate A07 (Markdown and LaTeX stripped)."""
    text = path.read_text(encoding="utf-8")
    a07 = plain(text[text.index("### A07"):text.index("### A08")])
    assert plain(QUESTION) in a07
    for title, _, lines, answer, _ in PANELS:
        joined = plain(" ".join(t for t, _ in lines))
        for piece in (p.strip() for p in joined.split("[...]")):
            for frag in re.split(r"(?<=\.) ", piece):  # sentences may be split across Markdown lines
                frag = frag.strip()
                if frag and frag.rstrip(" &") not in a07:
                    raise AssertionError(f"{title}: not verbatim in A07: {frag!r}")


def draw(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0.03, 1)
    ax.axis("off")
    line_h, pad = 0.036, 0.018
    y = 0.995
    ax.text(0.0, y, "Question", fontsize=7, color=MUTED, va="top", fontweight="bold")
    q1, q2 = textwrap.wrap(QUESTION + f"  (answer: {GOLD})", 62)
    ax.text(0.0, y - 0.042, q1, fontsize=7, color=INK, va="top")
    ax.text(0.0, y - 0.042 - line_h, q2, fontsize=7, color=INK, va="top")
    y -= 0.042 + 2 * line_h
    for k, sline in enumerate(textwrap.wrap("Solution: " + SOLUTION, 72)):
        ax.text(0.0, y - k * line_h, sline, fontsize=6.6, color=MUTED, va="top", style="italic")
    y -= line_h * (k + 1) + 0.03
    for title, color, lines, answer, right in PANELS:
        told = TOLD if color == STRONGER_REVIEWS else []
        h = 0.05 + line_h * (len(lines) + len(told)) + 0.022  # bottom padding 0.05 until 2026-10-06
        ax.add_patch(FancyBboxPatch((0.0, y - h), 1.0, h, boxstyle="round,pad=0,rounding_size=0.015",
                                    facecolor=SURFACE, edgecolor=color, linewidth=1.0, zorder=1))
        ax.text(0.025, y - pad, title, fontsize=7, color=INK, fontweight="bold", va="top", zorder=3)
        ax.text(0.975, y - pad, f"final answer {answer} " + ("✓" if right else "✗"), fontsize=7,
                color=RIGHT if right else WRONG, fontweight="bold", va="top", ha="right", zorder=3)
        ly = y - pad - 0.046
        for text in told:
            ax.text(0.025, ly, text, fontsize=6.4, color=MUTED, va="top", style="italic", zorder=3)
            ly -= line_h
        for text, mark in lines:
            t = ax.text(0.025, ly, text, fontsize=6.6, color=INK, va="top", family="DejaVu Sans Mono", zorder=3)
            if mark:
                t.set_bbox(dict(facecolor=MARK, edgecolor="none", pad=0.6))
            ly -= line_h
        y -= h + 0.03
    return y + 0.03


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", type=Path,
                    default=ROOT / "docs/analysis/example_candidates_20261004/candidates.md")
    ap.add_argument("--out", type=Path, default=ROOT / "docs/paper_draft/figures/fig2_example")
    args = ap.parse_args()
    check_verbatim(args.candidates)
    fig, ax = plt.subplots(figsize=(3.3, 3.25))
    fig.subplots_adjust(0.01, 0.01, 0.99, 0.99)
    bottom = draw(ax) - 0.01
    ax.set_ylim(bottom, 1)  # crop the empty space under the last box
    fig.set_size_inches(3.3, 3.25 * (1 - bottom) / 0.97)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(args.out.with_suffix("." + ext), dpi=300, bbox_inches="tight", facecolor="white")
    print(args.out.with_suffix(".png"))


if __name__ == "__main__":
    main()
