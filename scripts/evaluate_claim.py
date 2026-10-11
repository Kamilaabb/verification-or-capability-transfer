#!/usr/bin/env python3
"""Evaluate one claim card against a results database built by build_results_db.py.

Reads db/directions.jsonl and db/pairs.json (server only) and writes, into a new --out-dir,
reports/summary.md and reports/result.json. Both hold only counts and statistics, so they can be
committed. The primary analysis uses design pairs; "all pairs" is the sensitivity analysis.
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
from authority_allocation.claim_stats import Unit, claim_report  # noqa: E402


def stronger_generates_minus_stronger_reviews(rows, pairs, splits, design_only):
    """Unit per pair: a = final correctness with the stronger side generating, b = reversed."""
    gv = defaultdict(dict)
    for r in rows:
        if r["split"] in splits.get(r["domain"], []):
            gv[(r["domain"], r["generator"], r["reviewer"])][(r["split"], r["problem_id"])] = r["gv"]
    candidates, skipped = defaultdict(list), []
    for domain in splits:
        for p in pairs.get(domain, []):
            if design_only and not p["in_design_sample"]:
                continue
            label = f"{domain} {'/'.join(p['models'])}"
            if p["stronger"] is None:
                skipped.append(f"{label}: stronger side undetermined")
                continue
            s = p["stronger"]
            w = next(m for m in p["models"] if m != s)
            a, b = gv.get((domain, s, w), {}), gv.get((domain, w, s), {})
            if not a or not b:
                skipped.append(f"{label}: a direction is missing on the claim's splits")
                continue
            candidates[domain].append((s, w, a, b))
    # Each pair keeps all questions both of its directions answered; the bootstrap shares
    # question weights across pairs, so pairs run on different splits need not be cut down.
    units = []
    for domain, cands in candidates.items():
        for s, w, a, b in cands:
            keys = sorted(set(a) & set(b))
            if not keys:
                skipped.append(f"{domain} {s}/{w}: the two directions share no question")
                continue
            lost = (set(a) | set(b)) - set(keys)
            sizes = defaultdict(int)
            for split, _ in keys:
                sizes[split] += 1
            note = f"题目：{domain} {s}>{w} 用 {len(keys)} 道（{dict(sizes)}）"
            skipped.append(note + (f"，两方向不重合的 {len(lost)} 道不计入" if lost else ""))
            units.append(Unit(id=f"{domain}:{s}>{w}", domain=domain, a=[a[k] for k in keys],
                              b=[b[k] for k in keys], models=(s, w), keys=tuple(keys)))
    return units, skipped


def _directions(rows, pairs, splits, design_only, kind):
    """Yield one review direction at a time: (domain, generator, reviewer, kind, {key: (g, v, gv)}).

    kind is "stronger_reviews" when the reviewer is the pair's stronger side, else "weaker_reviews".
    Directions whose pair has no stronger side, or whose V solo is missing on any question, are skipped.
    """
    data = defaultdict(dict)
    for r in rows:
        if r["split"] in splits.get(r["domain"], []):
            data[(r["domain"], r["generator"], r["reviewer"])][(r["split"], r["problem_id"])] = (r["g"], r["v"], r["gv"])
    notes = []
    out = []
    for domain in splits:
        for p in pairs.get(domain, []):
            if design_only and not p["in_design_sample"]:
                continue
            for generator in p["models"]:
                reviewer = next(m for m in p["models"] if m != generator)
                d = data.get((domain, generator, reviewer))
                if not d:
                    continue
                if p["stronger"] is None:
                    notes.append(f"{domain} {reviewer} 审 {generator}: 较强方无法确定")
                    continue
                k = "stronger_reviews" if reviewer == p["stronger"] else "weaker_reviews"
                if kind not in ("all", k):
                    continue
                if any(v is None for _, v, _ in d.values()):
                    notes.append(f"{domain} {reviewer} 审 {generator}: 缺审阅者的独立作答，不计入")
                    continue
                out.append((domain, generator, reviewer, k, p, d))
    return out, notes


def _direction_units(rows, pairs, splits, design_only, kind, make):
    """Build one Unit per direction from make(g, v, gv, stronger_is_g) -> (a, b) or None (question left out)."""
    found, notes = _directions(rows, pairs, splits, design_only, kind)
    units = []
    for domain, generator, reviewer, k, p, d in found:
        keys, a, b = [], [], []
        for key in sorted(d):
            g, v, gv = d[key]
            ab = make(g, v, gv, p["stronger"] == generator)
            if ab is not None:
                keys.append(key)
                a.append(ab[0])
                b.append(ab[1])
        if not keys:
            notes.append(f"{domain} {reviewer} 审 {generator}: 没有符合条件的题")
            continue
        units.append(Unit(id=f"{domain}:{reviewer} 审 {generator}", domain=domain, a=a, b=b,
                          models=(generator, reviewer), keys=tuple(keys)))
        notes.append(f"题目：{domain} {reviewer} 审 {generator} 用 {len(keys)} 道")
    return units, notes


def gv_minus_g(rows, pairs, splits, design_only, kind="all"):
    return _direction_units(rows, pairs, splits, design_only, kind, lambda g, v, gv, sg: (gv, g))


def gv_minus_stronger_solo(rows, pairs, splits, design_only, kind="all"):
    return _direction_units(rows, pairs, splits, design_only, kind,
                            lambda g, v, gv, sg: (gv, g if sg else v))


def agreement_gain_minus_loss(rows, pairs, splits, design_only, kind="all"):
    return _direction_units(rows, pairs, splits, design_only, kind,
                            lambda g, v, gv, sg: (int(g == v == 0 and gv == 1), int(g == v == 1 and gv == 0)))


def follows_generator_on_disagreement(rows, pairs, splits, design_only, kind="all"):
    return _direction_units(rows, pairs, splits, design_only, kind,
                            lambda g, v, gv, sg: (int(gv == g), int(gv != g)) if g != v else None)


def exactly_one_right_before_minus_after(rows, pairs, splits, design_only, kind="all"):
    return _direction_units(rows, pairs, splits, design_only, kind,
                            lambda g, v, gv, sg: (int(g != v), int(g != gv)))


def oracle_minus_stronger_solo(rows, pairs, splits, design_only, kind="all"):
    """Pair-level: at least one of the two solo answers right, against the stronger solo alone."""
    found, notes = _directions(rows, pairs, splits, design_only, "all")
    units, done = [], set()
    for domain, generator, reviewer, k, p, d in found:
        key = (domain, frozenset(p["models"]))
        if key in done:
            continue
        done.add(key)
        keys = sorted(d)
        sg = p["stronger"] == generator
        units.append(Unit(id=f"{domain}:{'/'.join(sorted(p['models']))}", domain=domain,
                          a=[int(d[q][0] or d[q][1]) for q in keys],
                          b=[d[q][0] if sg else d[q][1] for q in keys],
                          models=tuple(p["models"]), keys=tuple(keys)))
    return units, notes


ESTIMANDS = {"stronger_generates_minus_stronger_reviews": stronger_generates_minus_stronger_reviews,
             "gv_minus_g": gv_minus_g,
             "gv_minus_stronger_solo": gv_minus_stronger_solo,
             "agreement_gain_minus_loss": agreement_gain_minus_loss,
             "follows_generator_on_disagreement": follows_generator_on_disagreement,
             "exactly_one_right_before_minus_after": exactly_one_right_before_minus_after,
             "oracle_minus_stronger_solo": oracle_minus_stronger_solo}

# Column labels: what a and b count, per question, in the plain words of the summary.
LABELS = {
    "stronger_generates_minus_stronger_reviews": ("较强者生成时对", "较强者审阅时对"),
    "gv_minus_g": ("审阅后对", "被审者独立作答对"),
    "gv_minus_stronger_solo": ("审阅后对", "较强者独立作答对"),
    "agreement_gain_minus_loss": ("两人都错、审阅后对", "两人都对、审阅后错"),
    "follows_generator_on_disagreement": ("审阅后对错跟被审者一样", "审阅后对错跟审阅者一样"),
    "exactly_one_right_before_minus_after": ("独立作答时一对一错", "审阅后与被审者一对一错"),
    "oracle_minus_stronger_solo": ("两人至少一个对", "较强者独立作答对"),
}
PAIR_LEVEL = {"stronger_generates_minus_stronger_reviews", "oracle_minus_stronger_solo"}
KIND_NAMES = {"all": "全部方向", "stronger_reviews": "较强者审较弱者", "weaker_reviews": "较弱者审较强者"}


def fmt_p(p):
    return f"{p:.3g}" if p >= 0.001 else "<0.001"


def render(card, analyses):
    a_label, b_label = LABELS[card["estimand_id"]]
    sign = card.get("expected_sign", "positive")
    lines = [f"# {card['id']}：{card['statement_zh']}", "",
             f"Δ = “{a_label}”的比例 − “{b_label}”的比例，单位百分点（pp）。"
             f"假说预期 Δ {'> 0' if sign == 'positive' else '< 0'}；“同向”指 Δ > 0。", ""]
    for name, (rep, skipped) in analyses.items():
        if rep is None:
            lines += [f"## {name}", ""] + [f"- {x}" for x in skipped] + [""]
            continue
        boot = rep["bootstrap"]
        lines += [f"## {name}", "", "|域|单位数|同向/反向/持平|符号检验 p|平均 Δ|95% 区间|",
                  "|---|---:|---:|---:|---:|---|"]
        for domain, st in rep["sign_test_per_domain"].items():
            b = boot["per_domain"][domain]
            lines.append(f"|{domain}|{st['units']}|{st['wins']}/{st['losses']}/{st['ties']}|{fmt_p(st['p_two_sided'])}|"
                         f"{b['mean_diff_pp']:+.2f}|[{b['ci95_pp'][0]:+.2f}, {b['ci95_pp'][1]:+.2f}]|")
        o, st, rp = boot["overall"], rep["sign_test"], rep["domain_replication"]
        lines += [f"|各域等权合计|{st['units']}|{st['wins']}/{st['losses']}/{st['ties']}|{fmt_p(st['p_two_sided'])}|"
                  f"{o['mean_diff_pp']:+.2f}|[{o['ci95_pp'][0]:+.2f}, {o['ci95_pp'][1]:+.2f}]|", "",
                  "区间：单位固定、每个域共用一次按题重抽（数学、知识按开发/测试分层，推理按 ReClor/FOLIO/GPQA 分层），"
                  "反映的是在这些固定的模型和配对下抽题带来的不确定性。"
                  + (f"敏感性（按模型对重抽，两个方向一起）：合计 [{rep['bootstrap_pairs']['overall']['ci95_pp'][0]:+.2f}, "
                     f"{rep['bootstrap_pairs']['overall']['ci95_pp'][1]:+.2f}]。" if "bootstrap_pairs" in rep else ""), "",
                  f"跨域：{rp['domains']} 个域中 {rp['positive']} 个同向、{rp['negative']} 个反向；"
                  f"区间不含 0 且同向的 {rp['ci_excludes_zero_positive']} 个，反向的 {rp['ci_excludes_zero_negative']} 个。", "",
                  "### 每个单位（区间为逐题配对 bootstrap）", "", f"|单位|{a_label}|{b_label}|只有前者|只有后者|Δ|95% 区间|McNemar p|",
                  "|---|---:|---:|---:|---:|---:|---|---:|"]
        for u in rep["units"]:
            lines.append(f"|{u['id']}|{u['a_correct']}/{u['n']}|{u['b_correct']}/{u['n']}|{u['a_only']}|{u['b_only']}|"
                         f"{u['diff_pp']:+.2f}|[{u['ci95_pp'][0]:+.2f}, {u['ci95_pp'][1]:+.2f}]|{fmt_p(u['mcnemar_p'])}|")
        lines += ["", "### 去掉一个模型后", "", "|去掉|剩余单位数|同向/反向|符号检验 p|平均 Δ|", "|---|---:|---:|---:|---:|"]
        for r in rep["leave_one_model_out"]:
            mean = "—" if r["mean_diff_pp"] is None else f"{r['mean_diff_pp']:+.2f}"
            lines.append(f"|{r['dropped_model']}|{r['units_left']}|{r['wins']}/{r['losses']}|{fmt_p(r['p_two_sided'])}|{mean}|")
        lines += ["", "### 未纳入的单位与每个单位用的题", ""] + ([f"- {s}" for s in skipped] or ["- 无"]) + [""]
    seen = card.get("seen_when_written", {})
    lines += ["## 写卡片时已看过的域", ""] + [f"- {d}: {v}" for d, v in seen.items()] + [
        "", "判定规则见卡片：逐域看平均 Δ 的 95% 区间；看过/没看过只作透明记录。所有域都已列出，不只列成立的。"]
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
    rows = [json.loads(line) for line in (args.db / "db/directions.jsonl").read_text().splitlines()]
    pairs = json.loads((args.db / "db/pairs.json").read_text())
    build = ESTIMANDS[card["estimand_id"]]
    boot = card.get("bootstrap", {})
    analyses = {}
    runs = [("主分析：设计配对", True, "all"), ("敏感性：全部已跑配对", False, "all")]
    if card["estimand_id"] not in PAIR_LEVEL:
        runs += [(f"分方向（设计配对）：{KIND_NAMES[k]}", True, k) for k in ("stronger_reviews", "weaker_reviews")]
    for name, design_only, kind in runs:
        if card["estimand_id"] in PAIR_LEVEL:
            units, skipped = build(rows, pairs, card["splits"], design_only)
        else:
            units, skipped = build(rows, pairs, card["splits"], design_only, kind)
        if not units:
            analyses[name] = (None, skipped + ["没有可用的单位，本节不计算"])
            continue
        analyses[name] = (claim_report(units, reps=boot.get("reps", 2000), seed=boot.get("seed", 20261003)), skipped)
    out = args.out_dir / "reports"
    out.mkdir(parents=True)
    (out / "summary.md").write_text(render(card, analyses), encoding="utf-8")
    (out / "result.json").write_text(json.dumps(
        {"claim": card["id"], "analyses": {k: {"report": r, "skipped": s} for k, (r, s) in analyses.items()},
         "db_pairs_sha256": hashlib.sha256((args.db / "db/pairs.json").read_bytes()).hexdigest(),
         "db_directions_sha256": hashlib.sha256((args.db / "db/directions.jsonl").read_bytes()).hexdigest(),
         "claim_sha256": hashlib.sha256(args.claim.read_bytes()).hexdigest(),
         "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(out / "summary.md")


if __name__ == "__main__":
    main()
