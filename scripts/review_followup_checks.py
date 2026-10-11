#!/usr/bin/env python3
"""Further checks (paper appendix, Further checks). Server only; no new inference.

    python scripts/review_followup_checks.py --db /path/to/scratch/results_db/20261007 \
        --artifacts /path/to/project/artifacts --out /path/to/scratch/claims/<new dir>

1. Copying G. For each design review direction, on the questions of
   Section 4 (H08 card splits): how often the final answer is the same as G's answer, against how often
   V's own solo answer is; and on the questions only V got right alone, of the final answers that are
   wrong, how many are G's (wrong) answer. "Same answer" is the same extracted answer string (math: the
   stored extracted_answer; knowledge and reasoning: the option letter; code: the final fenced program,
   code_pair_features.final_program). A missing answer is never the same as anything.
2. Truncation. Share of outputs that stopped at the 4,096-token limit (finish_reason == "length"):
   G alone, V alone, and the review output.
3. A selector without labels. Per design pair, on the questions of Section 5 (math and knowledge
   test1500, code and reasoning all), pick the answer whose output has the higher mean sampled-token
   log-probability (native logit trace, all positions); ties and missing traces go to the stronger model.
   Reported against the stronger model alone. "Same answer -> take it, else the stronger model" needs no
   check: it always returns the stronger model's answer.
4. Gap and headroom on test questions only. H08's regression rerun by explain_conditions.py with the
   card's splits restricted to test in math and knowledge (code and reasoning have no split).

Intervals: the audited bootstrap of claim_stats (directions fixed, one shared question resample per
domain within split / dataset strata, 2000 reps, seed 20261003). Means weight domains equally.
Writes a new directory, never overwrites; only aggregate counts and statistics leave the server.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from authority_allocation.claim_stats import Measured, bootstrap_statistic  # noqa: E402
from authority_allocation.code_pair_features import final_program  # noqa: E402
from gv_domain_tables import Archives  # noqa: E402

CARD = ROOT / "claims/H08_when_review_helps.yaml"
SECTION5 = {"math": ["test"], "knowledge": ["test"], "code": ["dev"], "nl_reasoning": ["dev"]}
KINDS = ("stronger_reviews", "weaker_reviews")


def read_jsonl(ar: Archives, run: Path, rel: str) -> list[dict]:
    return [json.loads(line) for line in ar.checked(run, rel).read_text().splitlines() if line.strip()]


def answer(domain: str, output: dict):
    if domain == "code":
        text = output.get("raw_output")
        return final_program(text) if text else None
    a = output.get("extracted_answer")
    return a.strip() if isinstance(a, str) and a.strip() else None


def truncated(output: dict) -> int:
    return int((output.get("usage") or {}).get("finish_reason") == "length")


def mean_logprob(trace: dict):
    if trace.get("status") not in (None, "ok"):
        return None
    lps = [p["sampled_logprob"] for p in trace.get("positions") or [] if p.get("sampled_logprob") is not None]
    return sum(lps) / len(lps) if lps else None


class Loader:
    """Per (domain, split, model, problem_id): the solo output and its mean log-probability, read from the
    review archives where that model is G (the review archive stores G's solo record byte-for-byte)."""

    def __init__(self, ar: Archives):
        self.ar, self.solo, self.lp, self.review, self.done = ar, {}, {}, {}, set()

    def load(self, domain: str, split: str, cid: str, generator: str, want_trace: bool) -> None:
        if (cid, want_trace) in self.done:
            return
        run = self.ar.find(cid)
        if run is None:
            raise ValueError(f"no archive for {cid}")
        # Keep only (answer, truncated): the raw outputs of 96 directions would not fit in memory.
        for r in read_jsonl(self.ar, run, "records/g_pass1.jsonl"):
            o = r["outputs"][0]
            self.solo[(domain, split, generator, str(r["problem_id"]))] = (answer(domain, o), truncated(o))
        for r in read_jsonl(self.ar, run, "records/review.jsonl"):
            o = r["outputs"][0]
            self.review[(cid, str(r["problem_id"]))] = (answer(domain, o), truncated(o))
        if want_trace:
            path = self.ar.checked(run, "records/g_native_logit_trace.jsonl")
            with path.open() as fh:
                for line in fh:
                    if line.strip():
                        t = json.loads(line)
                        self.lp[(domain, split, generator, str(t["problem_id"]))] = mean_logprob(t)
        self.done.add((cid, want_trace))


def ratio(a, b):
    return None if b == 0 else 100 * a / b


def equal_domain_mean(rows, key):
    per = defaultdict(list)
    for r in rows:
        if r.get(key) is not None:
            per[r["domain"]].append(r[key])
    return None if not per else sum(sum(v) / len(v) for v in per.values()) / len(per)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--artifacts", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; reports are never overwritten")
    card = yaml.safe_load(CARD.read_text())
    reps, seed = card["bootstrap"]["reps"], card["bootstrap"]["seed"]
    rows = [json.loads(line) for line in (args.db / "db/directions.jsonl").read_text().splitlines()]
    pairs = json.loads((args.db / "db/pairs.json").read_text())
    ar, loader = Archives(args.artifacts), Loader(Archives(args.artifacts))
    by_dir = defaultdict(dict)  # (domain, G, V) -> {(split, pid): (g, v, gv, condition)}
    for r in rows:
        by_dir[(r["domain"], r["generator"], r["reviewer"])][(r["split"], str(r["problem_id"]))] = (
            r["g"], r["v"], r["gv"], r["condition"])

    # ---------------------------------------------------------------- 1 and 2: per review direction
    items, kind_of, counts = [], {}, []
    for domain, splits in card["splits"].items():
        for p in pairs.get(domain, []):
            if not p["in_design_sample"]:
                continue
            for generator in p["models"]:
                reviewer = next(m for m in p["models"] if m != generator)
                d = {k: x for k, x in by_dir[(domain, generator, reviewer)].items() if k[0] in splits}
                if not d:
                    raise ValueError(f"{domain} {reviewer} reviews {generator}: no rows")
                rev = p["directions"][f"G={reviewer}"]
                for split, cid in p["directions"][f"G={generator}"].items():
                    if split in splits:
                        loader.load(domain, split, cid, generator, False)
                        loader.load(domain, split, rev[split], reviewer, False)
                keys = tuple(sorted(d))
                cols = defaultdict(list)
                for split, pid in keys:
                    g, v, gv, cid = d[(split, pid)]
                    if v is None:
                        raise ValueError(f"{domain} {reviewer} reviews {generator}: V solo missing")
                    ag, tg = loader.solo[(domain, split, generator, pid)]
                    av, tv = loader.solo[(domain, split, reviewer, pid)]
                    af, tf = loader.review[(cid, pid)]
                    same_fg = int(af is not None and af == ag)
                    same_vg = int(av is not None and av == ag)
                    only_v, only_g = int(g == 0 and v == 1), int(g == 1 and v == 0)
                    cols["same_final_g"].append(same_fg)
                    cols["same_v_g"].append(same_vg)
                    cols["only_v"].append(only_v)
                    cols["only_v_lost"].append(int(only_v and gv == 0))
                    cols["only_v_lost_to_g"].append(int(only_v and gv == 0 and same_fg))
                    cols["only_g"].append(only_g)
                    cols["only_g_lost"].append(int(only_g and gv == 0))
                    cols["only_g_lost_to_v"].append(int(only_g and gv == 0 and af is not None and af == av))
                    cols["trunc_g"].append(tg)
                    cols["trunc_v"].append(tv)
                    cols["trunc_review"].append(tf)
                uid = f"{domain}:{reviewer} reviews {generator}"
                kind = "stronger_reviews" if reviewer == p["stronger"] else "weaker_reviews"
                items.append(Measured(uid, domain, keys, dict(cols), models=(generator, reviewer)))
                kind_of[uid] = kind
                counts.append({"id": uid, "domain": domain, "kind": kind, "n": len(keys),
                               **{k: sum(c) for k, c in cols.items()}})
    found = {k: sum(1 for i in items if i.domain == k) for k in card["splits"]}
    if found != {k: 24 for k in card["splits"]}:
        raise ValueError(f"expected 24 design directions per domain, found {found}")

    def follow_stat(records):
        per = []
        for r in records:
            per.append({"domain": r["domain"], "kind": kind_of[r["id"]],
                        "same_final_g": r["same_final_g"], "same_v_g": r["same_v_g"],
                        "lost_to_g_share": ratio(r["only_v_lost_to_g"], r["only_v_lost"]),
                        "lost_to_v_share": ratio(r["only_g_lost_to_v"], r["only_g_lost"]),
                        "trunc_g": r["trunc_g"], "trunc_v": r["trunc_v"], "trunc_review": r["trunc_review"]})
        out = {}
        for kind in KINDS:
            sub = [x for x in per if x["kind"] == kind]
            for key in ("same_final_g", "same_v_g", "lost_to_g_share", "lost_to_v_share",
                        "trunc_g", "trunc_v", "trunc_review"):
                out[f"{kind}|{key}"] = equal_domain_mean(sub, key)
                for dom in sorted({x["domain"] for x in sub}):
                    vals = [x[key] for x in sub if x["domain"] == dom and x[key] is not None]
                    out[f"{kind}|{key}|{dom}"] = sum(vals) / len(vals) if vals else None
        return out

    follow = bootstrap_statistic(items, follow_stat, reps, seed, resample="fixed")["statistics"]

    # ---------------------------------------------------------------- 3: label-free selector
    pair_items, pair_counts = [], []
    for domain, splits in SECTION5.items():
        for p in pairs.get(domain, []):
            if not p["in_design_sample"]:
                continue
            a, b = p["models"]
            stronger = p["stronger"]
            weaker = b if stronger == a else a
            d = {k: x for k, x in by_dir[(domain, a, b)].items() if k[0] in splits}  # g = a, v = b
            for split in splits:
                loader.load(domain, split, p["directions"][f"G={a}"][split], a, True)
                loader.load(domain, split, p["directions"][f"G={b}"][split], b, True)
            keys = tuple(sorted(d))
            cols = defaultdict(list)
            for split, pid in keys:
                ua, ub = d[(split, pid)][0], d[(split, pid)][1]
                us, uw = (ua, ub) if stronger == a else (ub, ua)
                ls = loader.lp.get((domain, split, stronger, pid))
                lw = loader.lp.get((domain, split, weaker, pid))
                pick_w = int(ls is not None and lw is not None and lw > ls)
                pick = uw if pick_w else us
                cols["lp_minus_s"].append(pick - us)
                cols["pick_weaker"].append(pick_w)
                cols["missing_trace"].append(int(ls is None or lw is None))
                cols["oracle_minus_s"].append(int(uw == 1 and us == 0))
            uid = f"{domain}:{'/'.join(p['models'])}"
            pair_items.append(Measured(uid, domain, keys, dict(cols), models=tuple(p["models"])))
            pair_counts.append({"id": uid, "domain": domain, "n": len(keys), "stronger": stronger,
                                **{k: sum(c) for k, c in cols.items()}})

    def lp_stat(records):
        out = {}
        for key in ("lp_minus_s", "pick_weaker", "missing_trace", "oracle_minus_s"):
            out[key] = equal_domain_mean(records, key)
            for dom in sorted({r["domain"] for r in records}):
                vals = [r[key] for r in records if r["domain"] == dom]
                out[f"{key}|{dom}"] = sum(vals) / len(vals)
        return out

    lp = bootstrap_statistic(pair_items, lp_stat, reps, seed, resample="fixed")["statistics"]
    lp_pairs = {}
    for it in pair_items:
        s = bootstrap_statistic([it], lambda rs: {"d": rs[0]["lp_minus_s"]}, reps, seed, resample="fixed")
        lp_pairs[it.id] = s["statistics"]["d"]
    above = sum(1 for s in lp_pairs.values() if s["ci95"][0] > 0)
    below = sum(1 for s in lp_pairs.values() if s["ci95"][1] < 0)

    # ---------------------------------------------------------------- 4: H08 on test questions only
    args.out.mkdir(parents=True)
    test_card = dict(card, id="H08_test_only",
                     splits={d: (["test"] if "test" in s else s) for d, s in card["splits"].items()})
    card_path = args.out / "H08_test_only_card.yaml"
    card_path.write_text(yaml.safe_dump(test_card, allow_unicode=True, sort_keys=False), encoding="utf-8")
    subprocess.run([sys.executable, str(ROOT / "scripts/explain_conditions.py"), "--db", str(args.db),
                    "--claim", str(card_path), "--out-dir", str(args.out / "H08_test_only")], check=True)

    result = {"db": str(args.db), "card": str(CARD), "reps": reps, "seed": seed,
              "follow_and_truncation": follow, "direction_counts": counts,
              "label_free_selector": lp, "label_free_pairs": lp_pairs,
              "label_free_pairs_above_below": [above, below], "pair_counts": pair_counts}
    (args.out / "reports").mkdir()
    (args.out / "reports/result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n",
                                                  encoding="utf-8")

    def fmt(s):
        return "—" if s is None or s["point"] is None else (
            f"{s['point']:.1f} [{s['ci95'][0]:.1f}, {s['ci95'][1]:.1f}]" if s["ci95"] else f"{s['point']:.1f}")

    lines = ["# 审稿意见要的四项检查（2026-10-09）", "",
             "不新跑模型，只读已有存档和结果库。区间：方向固定、每个域共用一次按题重抽（开发/测试分层，推理按数据集分层），"
             f"{reps} 次，种子 {seed}；平均四域等权。单位：百分比。", "",
             "## 1. 最终答案和 G 的答案相同（第 4 节的题）", "",
             "|谁审谁|最终 = G|V 单独 = G|只有 V 对的题上丢掉的里面，变成 G 的答案|只有 G 对的题上丢掉的里面，变成 V 自己的答案|",
             "|---|---|---|---|---|"]
    for kind in KINDS:
        lines.append(f"|{kind}|{fmt(follow[f'{kind}|same_final_g'])}|{fmt(follow[f'{kind}|same_v_g'])}|"
                     f"{fmt(follow[f'{kind}|lost_to_g_share'])}|{fmt(follow[f'{kind}|lost_to_v_share'])}|")
    lines += ["", "逐域（点估计）：", "", "|谁审谁|域|最终 = G|V 单独 = G|丢掉的变成 G 的答案|", "|---|---|---|---|---|"]
    for kind in KINDS:
        for dom in card["splits"]:
            lines.append(f"|{kind}|{dom}|{fmt(follow.get(f'{kind}|same_final_g|{dom}'))}|"
                         f"{fmt(follow.get(f'{kind}|same_v_g|{dom}'))}|{fmt(follow.get(f'{kind}|lost_to_g_share|{dom}'))}|")
    lines += ["", "## 2. 截断率（输出停在 4,096 token）", "", "|谁审谁|G 单独|V 单独|审阅输出|", "|---|---|---|---|"]
    for kind in KINDS:
        lines.append(f"|{kind}|{fmt(follow[f'{kind}|trunc_g'])}|{fmt(follow[f'{kind}|trunc_v'])}|"
                     f"{fmt(follow[f'{kind}|trunc_review'])}|")
    lines += ["", "逐域（点估计）：", "", "|谁审谁|域|G 单独|V 单独|审阅输出|", "|---|---|---|---|---|"]
    for kind in KINDS:
        for dom in card["splits"]:
            lines.append(f"|{kind}|{dom}|{fmt(follow.get(f'{kind}|trunc_g|{dom}'))}|"
                         f"{fmt(follow.get(f'{kind}|trunc_v|{dom}'))}|{fmt(follow.get(f'{kind}|trunc_review|{dom}'))}|")
    lines += ["", "## 3. 不用标签的选择器：选平均 log-prob 高的答案（第 5 节的题）", "",
              f"相对较强者单独：{fmt(lp['lp_minus_s'])}；选了较弱者的比例 {fmt(lp['pick_weaker'])}；"
              f"上限空间 {fmt(lp['oracle_minus_s'])}；缺 trace 的比例 {fmt(lp['missing_trace'])}。"
              f"48 对里显著高于较强者 {above} 对、低于 {below} 对。", "",
              "|域|相对较强者|选较弱者的比例|", "|---|---|---|"]
    for dom in SECTION5:
        lines.append(f"|{dom}|{fmt(lp.get(f'lp_minus_s|{dom}'))}|{fmt(lp.get(f'pick_weaker|{dom}'))}|")
    lines += ["", "## 4. 只用测试集的差距 / 上限空间回归", "",
              "见 `H08_test_only/reports/summary.md`（数学、知识只用 test1500；代码、推理没有拆分，和原来一样）。", ""]
    (args.out / "reports/summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.out / "reports/summary.md")


if __name__ == "__main__":
    main()
