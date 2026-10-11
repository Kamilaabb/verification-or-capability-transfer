#!/usr/bin/env python3
"""When does review raise the generator, and when does it beat the stronger model alone?

Reads a results database built by build_results_db.py (server only) and a conditions card
(claims/H08_*.yaml), and writes into a new --out-dir reports/summary.md and reports/result.json.
Both hold only counts and statistics, so they can be committed.

For every review direction it measures, on that direction's own questions:

  Δ_G       acc(after review) − acc(generator alone)
  Δ_C       acc(after review) − acc(stronger model of the pair alone)
  gap       acc(stronger alone) − acc(weaker alone)
  headroom  acc(at least one of the two right alone) − acc(stronger alone)
            = share of questions the weaker gets right and the stronger gets wrong
  disagree  share of questions exactly one of the two gets right

and then (1) sorts every direction into: beats the stronger model / raises the generator but
not past the stronger model / does not raise the generator, (2) relates Δ_G and Δ_C to gap and
headroom across directions, separately for "stronger reviews weaker" and "weaker reviews
stronger", with one intercept per domain, and (3) relates gap to headroom across pairs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from authority_allocation.claim_stats import (  # noqa: E402
    Measured, Unit, bootstrap_statistic, fe_correlation, fe_slopes, paired_unit)
from evaluate_claim import KIND_NAMES, _directions, fmt_p  # noqa: E402

TIER_ORDER = {"strong": 0, "middle": 1, "weak": 2}
TIER_ZH = {"strong": "强", "middle": "中", "weak": "弱"}
CELL_ORDER = ["strong-strong", "strong-middle", "strong-weak", "middle-middle", "middle-weak", "weak-weak"]
CATEGORY = {"green": "🟢 超过较强者", "yellow": "🟡 提升了被审者、没超过较强者", "white": "⚪ 没提升被审者"}
KINDS = ("stronger_reviews", "weaker_reviews")
OUTCOMES = {"dC": "Δ_C", "dG": "Δ_G"}
FACTORS = {"gap": "能力差距", "head": "上限空间"}
SPECS = {"gap": ["gap"], "head": ["head"], "both": ["gap", "head"]}
CARD_KEYS = {"gap": "gap", "head": "headroom"}  # factor names as the card's `expected` spells them
SPEC_ZH = {"gap": "只看差距", "head": "只看空间", "both": "两者一起"}


def cell_of(pair):
    if pair.get("cell"):
        return pair["cell"]
    tiers = pair.get("tiers") or [None, None]
    if None in tiers:
        return None
    return "-".join(sorted(tiers, key=TIER_ORDER.get))


def cell_zh(cell):
    if not cell:
        return "—"
    parts = cell.split("-")
    return "".join(TIER_ZH[t] for t in parts) if all(t in TIER_ZH for t in parts) else cell


def measure(rows, pairs, splits, design_only):
    """One Measured per direction and one per pair, plus the per-direction category."""
    found, notes = _directions(rows, pairs, splits, design_only, "all")
    directions, by_pair = [], defaultdict(dict)
    for domain, generator, reviewer, kind, p, d in found:
        sg = p["stronger"] == generator
        keys = tuple(sorted(d))
        s = [d[k][0] if sg else d[k][1] for k in keys]
        w = [d[k][1] if sg else d[k][0] for k in keys]
        g = [d[k][0] for k in keys]
        gv = [d[k][2] for k in keys]
        cols = {"dG": [a - b for a, b in zip(gv, g)], "dC": [a - b for a, b in zip(gv, s)],
                "gap": [a - b for a, b in zip(s, w)], "head": [int(b == 1 and a == 0) for a, b in zip(s, w)],
                "dis": [int(a != b) for a, b in zip(s, w)]}
        uid = f"{domain}:{reviewer} 审 {generator}"
        item = Measured(uid, domain, keys, cols, models=(generator, reviewer))
        # Keys carry the question strata into the paired bootstrap (statistics audit, 2026-10-05).
        dg = paired_unit(Unit(uid, domain, gv, g, keys=keys))
        dc = paired_unit(Unit(uid, domain, gv, s, keys=keys))
        if dc["ci95_pp"][0] > 0:
            category = "green"
        elif dg["ci95_pp"][0] > 0:
            category = "yellow"
        else:
            category = "white"
        directions.append({"item": item, "kind": kind, "category": category, "worse": dg["ci95_pp"][1] < 0,
                           "dG": dg, "dC": dc, "cell": cell_of(p), "headroom_label": p.get("headroom"),
                           "pair": "/".join(p["models"]), "stronger": p["stronger"], "n": len(keys)})
        solos = by_pair[(domain, tuple(p["models"]), p["stronger"], cell_of(p), p.get("headroom"))]
        for k, a, b in zip(keys, s, w):
            solos.setdefault(k, (a, b))
    pair_items = []
    for (domain, models, stronger, cell, label), solos in by_pair.items():
        keys = tuple(sorted(solos))
        s = [solos[k][0] for k in keys]
        w = [solos[k][1] for k in keys]
        item = Measured(f"{domain}:{'/'.join(models)}", domain, keys,
                        {"gap": [a - b for a, b in zip(s, w)], "head": [int(b == 1 and a == 0) for a, b in zip(s, w)],
                         "dis": [int(a != b) for a, b in zip(s, w)]}, models=models)
        pair_items.append({"item": item, "cell": cell, "headroom_label": label})
    return directions, pair_items, notes


def slope_statistic(records):
    """Δ_C and Δ_G against gap and headroom, pooled (domain intercepts) and per domain."""
    out = {}
    scopes = [("合计", records)] + [(d, [r for r in records if r["domain"] == d])
                                    for d in sorted({r["domain"] for r in records})]
    for scope, rs in scopes:
        for outcome in OUTCOMES:
            for spec, xs in SPECS.items():
                fit = fe_slopes(rs, outcome, xs) if len(rs) > len(xs) + 1 else None
                for x in xs:
                    out[f"{scope}|{outcome}|{spec}|{x}"] = None if fit is None else 10 * fit[x]
    return out


def correlation_statistic(records):
    out = {"合计": fe_correlation(records, "gap", "head")}
    for d in sorted({r["domain"] for r in records}):
        rs = [r for r in records if r["domain"] == d]
        out[d] = fe_correlation(rs, "gap", "head") if len(rs) > 2 else None
    return out


def leave_one_model_out(records):
    rows = []
    for model in sorted({m for r in records for m in r["models"]}):
        kept = [r for r in records if model not in r["models"]]
        fits = {o: fe_slopes(kept, o, SPECS["both"]) if len(kept) > 3 else None for o in OUTCOMES}
        rows.append({"dropped_model": model, "units_left": len(kept),
                     **{f"{o}|{x}": None if fits[o] is None else 10 * fits[o][x]
                        for o in OUTCOMES for x in SPECS["both"]}})
    return rows


def analyse(rows, pairs, card, design_only):
    directions, pair_items, notes = measure(rows, pairs, card["splits"], design_only)
    boot = card.get("bootstrap", {})
    reps, seed = boot.get("reps", 2000), boot.get("seed", 20261003)
    result = {"directions": [], "pairs": [], "notes": notes, "fits": {}, "lomo": {}, "correlation": None}
    for d in directions:
        rec = d["item"].record()
        result["directions"].append({"id": d["item"].id, "domain": d["item"].domain, "kind": d["kind"],
                                     "pair": d["pair"], "stronger": d["stronger"], "cell": d["cell"],
                                     "headroom_label": d["headroom_label"], "n": d["n"],
                                     "gap": rec["gap"], "head": rec["head"], "dis": rec["dis"],
                                     "dG": rec["dG"], "dG_ci95": d["dG"]["ci95_pp"],
                                     "dG_ci95_pre_audit": d["dG"]["ci95_pp_pre_audit"],
                                     "dC": rec["dC"], "dC_ci95": d["dC"]["ci95_pp"],
                                     "dC_ci95_pre_audit": d["dC"]["ci95_pp_pre_audit"],
                                     "category": d["category"], "worse": d["worse"]})
    for p in pair_items:
        result["pairs"].append({"id": p["item"].id, "domain": p["item"].domain, "cell": p["cell"],
                                "headroom_label": p["headroom_label"], **{k: p["item"].value(k) for k in ("gap", "head", "dis")}})
    if pair_items:
        result["correlation"] = bootstrap_statistic([p["item"] for p in pair_items], correlation_statistic, reps, seed)
    for kind in KINDS:
        items = [d["item"] for d in directions if d["kind"] == kind]
        if not items:
            continue
        result["fits"][kind] = bootstrap_statistic(items, slope_statistic, reps, seed)
        result["lomo"][kind] = leave_one_model_out([u.record() for u in items])
    return result


def ci_text(s):
    if s is None or s["point"] is None:
        return "无法估计", "—", "—"
    ci = "—" if s["ci95"] is None else f"[{s['ci95'][0]:+.2f}, {s['ci95'][1]:+.2f}]"
    return f"{s['point']:+.2f}", ci, "—" if s["p_two_sided"] is None else fmt_p(s["p_two_sided"])


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def render_analysis(name, res, card):
    lines = [f"## {name}", ""]
    dirs = res["directions"]
    if not dirs:
        return lines + ["没有可用的审阅方向，本节不计算。", ""] + [f"- {n}" for n in res["notes"]] + [""]
    domains = sorted({d["domain"] for d in dirs})

    lines += ["### 1. 每个审阅方向属于哪一类", "",
              "分类看配对差的 95% 区间：Δ_C 区间整个在 0 以上 → 🟢；否则 Δ_G 区间整个在 0 以上 → 🟡；其余 → ⚪"
              "（其中 Δ_G 区间整个在 0 以下的标“变差”）。较弱者审较强者时被审者就是较强者，Δ_G 与 Δ_C 相同，所以没有 🟡。", "",
              "|域|方向|🟢|🟡|⚪（其中变差）|", "|---|---|---:|---:|---:|"]
    for domain in domains:
        for kind in KINDS:
            ds = [d for d in dirs if d["domain"] == domain and d["kind"] == kind]
            if ds:
                c = {k: sum(d["category"] == k for d in ds) for k in CATEGORY}
                lines.append(f"|{domain}|{KIND_NAMES[kind]}|{c['green']}|{c['yellow']}|"
                             f"{c['white']}（{sum(d['worse'] for d in ds)}）|")
    lines += ["", "|域|方向|配对（格，空间高低）|题数|差距|上限空间|Δ_G [95% 区间]|Δ_C [95% 区间]|类别|",
              "|---|---|---|---:|---:|---:|---|---|---|"]
    for domain in domains:
        for kind in KINDS:
            for d in sorted((d for d in dirs if d["domain"] == domain and d["kind"] == kind), key=lambda d: -d["gap"]):
                label = {"high": "高", "low": "低"}.get(d["headroom_label"], "—")
                tag = CATEGORY[d["category"]].split()[0] + ("（变差）" if d["worse"] else "")
                lines.append(f"|{domain}|{KIND_NAMES[kind]}|{d['id'].split(':', 1)[1]}（{cell_zh(d['cell'])}，{label}）|{d['n']}|"
                             f"{d['gap']:.1f}|{d['head']:.1f}|{d['dG']:+.2f} [{d['dG_ci95'][0]:+.2f}, {d['dG_ci95'][1]:+.2f}]|"
                             f"{d['dC']:+.2f} [{d['dC_ci95'][0]:+.2f}, {d['dC_ci95'][1]:+.2f}]|{tag}|")

    lines += ["", "### 2. 按格子汇总（只作描述：每格每域只有一两对，不做检验）", "",
              "|格|方向|方向数|平均差距|平均上限空间|平均 Δ_G|平均 Δ_C|🟢/🟡/⚪|", "|---|---|---:|---:|---:|---:|---:|---|"]
    cells = sorted({d["cell"] for d in dirs}, key=lambda c: CELL_ORDER.index(c) if c in CELL_ORDER else 99)
    for cell in cells:
        for kind in KINDS:
            ds = [d for d in dirs if d["cell"] == cell and d["kind"] == kind]
            if ds:
                c = {k: sum(d["category"] == k for d in ds) for k in CATEGORY}
                lines.append(f"|{cell_zh(cell)}|{KIND_NAMES[kind]}|{len(ds)}|{mean(d['gap'] for d in ds):.1f}|"
                             f"{mean(d['head'] for d in ds):.1f}|{mean(d['dG'] for d in ds):+.2f}|{mean(d['dC'] for d in ds):+.2f}|"
                             f"{c['green']}/{c['yellow']}/{c['white']}|")

    lines += ["", "### 3. 能力差距和上限空间的关系（每对一个点）", "",
              "恒等式：上限空间 = (分歧率 − 能力差距) / 2，分歧率 = 一对一错的题的比例。所以分歧率差不多时，"
              "差距越大、空间越小；但分歧率本身随配对变，两者并不是同一个量。下面的相关系数去掉了各域的平均水平。", "",
              "|范围|对数|相关系数|95% 区间|p|平均差距|平均上限空间|平均分歧率|", "|---|---:|---:|---|---:|---:|---:|---:|"]
    stats = res["correlation"]["statistics"]
    for scope in ["合计", *sorted({p["domain"] for p in res["pairs"]})]:
        ps = res["pairs"] if scope == "合计" else [p for p in res["pairs"] if p["domain"] == scope]
        point, ci, p = ci_text(stats.get(scope))
        lines.append(f"|{scope}|{len(ps)}|{point}|{ci}|{p}|{mean(x['gap'] for x in ps):.1f}|"
                     f"{mean(x['head'] for x in ps):.1f}|{mean(x['dis'] for x in ps):.1f}|")

    expected = card.get("expected", {})
    lines += ["", "### 4. 差距、上限空间各自和 Δ 的关系", "",
              "斜率 = 差距（或上限空间）每大 10 个百分点，Δ 平均变多少个百分点；每个域有自己的截距，"
              "所以比较的是同一个域内的配对；各域等权。“两者一起”是控制另一个因素之后的斜率，"
              "卡片的预期针对它；“只看”一栏供对照。区间来自按模型对重抽（两个方向一起）加按题重抽的 bootstrap；只描述这些配对上的线性关系。", ""]
    for kind in KINDS:
        fit = res["fits"].get(kind)
        if fit is None:
            continue
        lines += [f"#### {KIND_NAMES[kind]}", "", "|结果|模型|因素|预期|斜率（每 10pp）|95% 区间|p|",
                  "|---|---|---|---|---:|---|---:|"]
        outcomes = ["dC"] if kind == "weaker_reviews" else list(OUTCOMES)
        for outcome in outcomes:
            for spec, xs in SPECS.items():
                for x in xs:
                    exp = expected.get(kind, {}).get(OUTCOMES[outcome], {}).get(CARD_KEYS[x]) if spec == "both" else None
                    exp = {"positive": "> 0", "negative": "< 0"}.get(exp, "—")
                    point, ci, p = ci_text(fit["statistics"][f"合计|{outcome}|{spec}|{x}"])
                    lines.append(f"|{OUTCOMES[outcome]}|{SPEC_ZH[spec]}|{FACTORS[x]}|{exp}|{point}|{ci}|{p}|")
        if kind == "weaker_reviews":
            lines.append("")
            lines.append("（较弱者审较强者时 Δ_G = Δ_C，只列一次。）")
        lines += ["", "逐域（两者一起，只作描述）：", "", "|域|结果|差距斜率 [95% 区间]|上限空间斜率 [95% 区间]|",
                  "|---|---|---|---|"]
        for domain in domains:
            for outcome in outcomes:
                cells_ = []
                for x in SPECS["both"]:
                    point, ci, _ = ci_text(fit["statistics"].get(f"{domain}|{outcome}|both|{x}"))
                    cells_.append(f"{point} {ci}" if ci != "—" else point)
                lines.append(f"|{domain}|{OUTCOMES[outcome]}|{cells_[0]}|{cells_[1]}|")
        lines += ["", "去掉一个模型后（两者一起，合计，只列点估计）：", "",
                  "|去掉|剩余方向数|" + "|".join(f"{OUTCOMES[o]}~{FACTORS[x]}" for o in outcomes for x in SPECS["both"]) + "|",
                  "|---|---:|" + "---:|" * (len(outcomes) * 2)]
        for r in res["lomo"][kind]:
            vals = ["无法估计" if r[f"{o}|{x}"] is None else f"{r[f'{o}|{x}']:+.2f}" for o in outcomes for x in SPECS["both"]]
            lines.append(f"|{r['dropped_model']}|{r['units_left']}|" + "|".join(vals) + "|")
        lines.append("")
    lines += ["### 未纳入的方向与每个方向用的题", ""] + ([f"- {n}" for n in res["notes"]] or ["- 无"]) + [""]
    return lines


def render(card, analyses):
    lines = [f"# {card['id']}：{card['statement_zh']}", "",
             "- Δ_G = 审阅后准确率 − 被审者自己的准确率；Δ_C = 审阅后准确率 − 较强者自己的准确率。",
             "- 能力差距 = 较强者自己的准确率 − 较弱者自己的准确率；上限空间 = 两人至少一个对的比例 − 较强者自己对的比例，"
             "也就是较弱者对、较强者错的题的比例。",
             "- 较强者审较弱者时有恒等式 Δ_G = 能力差距 + Δ_C：被审者被拉高的量，一部分只是把较强者的水平转了过来。",
             "- 所有量都在该方向自己的题上算，单位百分点（pp）。", ""]
    for name, res in analyses.items():
        lines += render_analysis(name, res, card)
    seen = card.get("seen_when_written", {})
    lines += ["## 写卡片时已看过的域", ""] + [f"- {d}: {v}" for d, v in seen.items()] + [
        "", "判定规则见卡片；所有域、所有方向都已列出，不只列成立的。"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, required=True, help="out-dir of build_results_db.py")
    ap.add_argument("--claim", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    if args.out_dir.exists():
        raise SystemExit(f"{args.out_dir} exists; claim reports are never overwritten")
    card = yaml.safe_load(args.claim.read_text())
    if card.get("analysis") != "conditions":
        raise SystemExit(f"{args.claim} is not a conditions card; use evaluate_claim.py")
    rows = [json.loads(line) for line in (args.db / "db/directions.jsonl").read_text().splitlines()]
    pairs = json.loads((args.db / "db/pairs.json").read_text())
    analyses = {"主分析：设计配对": analyse(rows, pairs, card, True),
                "敏感性：全部已跑配对": analyse(rows, pairs, card, False)}
    out = args.out_dir / "reports"
    out.mkdir(parents=True)
    (out / "summary.md").write_text(render(card, analyses), encoding="utf-8")
    (out / "result.json").write_text(json.dumps(
        {"claim": card["id"], "analyses": analyses,
         "db_pairs_sha256": hashlib.sha256((args.db / "db/pairs.json").read_bytes()).hexdigest(),
         "db_directions_sha256": hashlib.sha256((args.db / "db/directions.jsonl").read_bytes()).hexdigest(),
         "claim_sha256": hashlib.sha256(args.claim.read_bytes()).hexdigest(),
         "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(out / "summary.md")


if __name__ == "__main__":
    main()
