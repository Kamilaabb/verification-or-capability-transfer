"""Per-question correctness and token counts of every RQ3/RQ4 system, for the design pairs (server only).

    python scripts/build_cost_rows.py --db /path/to/scratch/results_db/20261004b \
        --artifacts artifacts --out-dir /path/to/scratch/cost/<date>

Reads the results database (db/*.jsonl, db/pairs.json, reports/systems.json) for correctness and the
archives it names for token counts. Each design pair is compared on the questions its selector was
evaluated on: the held-out test split for math and knowledge, every question (cross-fitted) for code and
reasoning. A review is charged for the generator answer it actually read, from its own source archive.

  db/cost_rows.jsonl       one row per pair and question: correctness of every system and tokens of
                           every call (problem IDs; stays on the server)
  reports/cost_pairs.json  per pair: accuracy, mean tokens per call, cost per question (safe to commit)
  reports/summary.md       the same in a table, plus every pair or archive left out and why

Nothing is guessed: a missing or ambiguous archive, a record without token counts, or a question one
system lacks drops the pair and is listed in the summary.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from authority_allocation.cost_model import (CALL_MODEL, JEV_USD_PER_M_INPUT, OUTPUT_WEIGHTS,  # noqa: E402
                                             PARAMS_B, SYSTEMS, system_cost, usage_of)

_spec = importlib.util.spec_from_file_location("gv_domain_tables", ROOT / "scripts/gv_domain_tables.py")
_gv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gv)
Archives = _gv.Archives

SOLO_FILES = ("records/pass1.jsonl", "records/g.jsonl")
REVIEW_FILES = ("records/review.jsonl", "records/v.jsonl")


def listed(run: Path) -> set[str]:
    return {line.split(maxsplit=1)[1].lstrip("*").removeprefix("./")
            for line in (run / "checksums.sha256").read_text().splitlines() if line.strip()}


def read_usage(ar: Archives, run: Path, names: tuple[str, ...]) -> dict[str, tuple[int, int]]:
    """Token counts per question from the first of `names` the archive's checksums list."""
    rel = next((n for n in names if n in listed(run)), None)
    if rel is None:
        raise ValueError(f"{run.name}: none of {names} in checksums")
    out = {}
    for line in ar.checked(run, rel).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("sample_index", 0) != 0:
            continue  # sample_index 0 is y0
        pid = str(r["problem_id"])
        if pid in out:
            raise ValueError(f"{run.name}/{rel}: duplicate problem {pid}")
        out[pid] = usage_of(r)
    return out


def jev_tokens(source: Path, ids: set[str], sources: dict) -> dict:
    files = sorted(source.rglob("jev_responses.jsonl"))
    if not files:
        raise ValueError(f"{source}: no jev_responses.jsonl")
    calls = {}
    for path in files:
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        for line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            pid = str(r.get("problem_id"))
            if pid not in ids:
                continue
            usage = (r.get("response") or {}).get("usage") or {}
            tok = (usage.get("input_tokens"), usage.get("output_tokens", 0))
            if any(type(x) is not int for x in tok):
                raise ValueError(f"{path}: JEV call without token counts for {pid}")
            key = (pid, r.get("order"))
            if key in calls and calls[key] != tok:
                raise ValueError(f"{path}: two different token counts for {key}")
            calls[key] = tok
    inp = sum(t[0] for t in calls.values())
    return {"calls": len(calls), "questions_called": len({k[0] for k in calls}), "input_tokens": inp,
            "output_tokens": sum(t[1] for t in calls.values()), "usd_total": inp * JEV_USD_PER_M_INPUT / 1e6,
            "usd_per_1k_questions": inp * JEV_USD_PER_M_INPUT / 1e6 / len(ids) * 1000}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build(db: Path, artifacts: Path) -> dict:
    ar = Archives(artifacts)
    pairs = json.loads((db / "db/pairs.json").read_text(encoding="utf-8"))
    summaries = json.loads((db / "reports/systems.json").read_text(encoding="utf-8"))["systems"]
    solo, solo_cond = defaultdict(dict), defaultdict(set)
    for r in load_jsonl(db / "db/solo.jsonl"):
        k = (r["domain"], r["split"], r["model"])
        solo[k][str(r["problem_id"])] = r["correct"]
        solo_cond[k].add(r["condition"])
    dirs, dir_cond = defaultdict(dict), defaultdict(set)
    for r in load_jsonl(db / "db/directions.jsonl"):
        k = (r["domain"], r["split"], r["generator"], r["reviewer"])
        dirs[k][str(r["problem_id"])] = r["gv"]
        dir_cond[k].add(r["condition"])
    systems, sys_split = defaultdict(dict), defaultdict(set)
    for r in load_jsonl(db / "db/systems.jsonl"):
        systems[(r["domain"], r["pair_id"], r["system"])][str(r["problem_id"])] = r["correct"]
        sys_split[(r["domain"], r["pair_id"])].add((r["split"], r["evaluation"]))
    upstream = {(s["domain"], "|".join(s["models"]), s["system"]): s for s in summaries}

    rows, reports, problems, notes = [], [], [], []
    for domain, plist in pairs.items():
        for p in plist:
            if not p["in_design_sample"]:
                continue
            a, b = p["models"]
            pair_id = "|".join(sorted((a, b)))
            tag = f"{domain} {a} / {b}"
            try:
                s = p["stronger"]
                if s is None:
                    raise ValueError("stronger side unknown")
                w = b if s == a else a
                if {s, w} - PARAMS_B.keys():
                    raise ValueError(f"no parameter count for {sorted({s, w} - PARAMS_B.keys())}")
                splits = sys_split[(domain, pair_id)]
                if len(splits) != 1:
                    raise ValueError(f"selector/JEV splits {sorted(splits)}")
                split, evaluation = next(iter(splits))

                def one(index, key, what):
                    conds = index[key]
                    if len(conds) != 1:
                        raise ValueError(f"{what}: conditions {sorted(conds)}")
                    run = ar.find(next(iter(conds)))
                    if run is None:
                        raise ValueError(f"{what}: archive of {next(iter(conds))} not found")
                    return run

                correct = {"weaker_alone": solo[(domain, split, w)], "stronger_alone": solo[(domain, split, s)],
                           "review_by_stronger": dirs[(domain, split, w, s)],
                           "review_by_weaker": dirs[(domain, split, s, w)]}
                for name in ("selector_primary", "selector_secondary", "JEV"):
                    correct[name] = systems[(domain, pair_id, name)]
                ids = set(correct["selector_primary"])
                short = [k for k, v in correct.items() if not ids <= set(v)]
                if not ids or short:
                    raise ValueError(f"systems missing questions of the selector's set: {short}")
                runs = {"solo_w": one(solo_cond, (domain, split, w), f"{w} alone"),
                        "solo_s": one(solo_cond, (domain, split, s), f"{s} alone"),
                        "review_by_s": one(dir_cond, (domain, split, w, s), f"{s} reviews"),
                        "review_by_w": one(dir_cond, (domain, split, s, w), f"{w} reviews")}
                for call, review in (("g_of_review_by_s", "review_by_s"), ("g_of_review_by_w", "review_by_w")):
                    src = (ar.config(runs[review]).get("generator_source") or {}).get("artifact_path")
                    g_run = artifacts / Path(src).name if src else None
                    if g_run is None or not (g_run / "checksums.sha256").exists():
                        raise ValueError(f"{runs[review].name}: generator source archive missing")
                    runs[call] = g_run
                tokens = {c: read_usage(ar, r, REVIEW_FILES if c.startswith("review") else SOLO_FILES)
                          for c, r in runs.items()}
                lacking = [c for c, t in tokens.items() if not ids <= set(t)]
                if lacking:
                    raise ValueError(f"token counts missing for some questions: {lacking}")
                for name in ("selector_primary", "JEV"):
                    up = upstream.get((domain, pair_id, name), {}).get("upstream_runs", [])
                    for c in ("solo_w", "solo_s"):
                        if runs[c].name not in up:
                            notes.append(f"{tag}: {name} 的上游单独作答 {up} 不含 {runs[c].name}（准确率仍以结果库为准）")
                jev_src = upstream.get((domain, pair_id, "JEV"), {}).get("source")
                if not jev_src:
                    raise ValueError("JEV source not in reports/systems.json")
                jev = jev_tokens(Path(jev_src), ids, ar.sources)
            except (ValueError, KeyError) as e:
                problems.append(f"{tag}: {e}")
                continue
            for c in ("g_of_review_by_s", "g_of_review_by_w"):
                own = "solo_w" if c == "g_of_review_by_s" else "solo_s"
                differ = sum(tokens[c][q] != tokens[own][q] for q in ids)
                if differ:
                    notes.append(f"{tag}: 审阅读到的 G 答案与单独作答归档 token 数不同的题 {differ}/{len(ids)}")
            params = {"w": PARAMS_B[w], "s": PARAMS_B[s]}
            pair_rows = []
            for q in sorted(ids):
                c = {k: int(v[q]) for k, v in correct.items()}
                c["oracle"] = max(c["weaker_alone"], c["stronger_alone"])
                pair_rows.append({"domain": domain, "split": split, "evaluation": evaluation, "pair_id": pair_id,
                                  "stronger": s, "weaker": w, "params_b": params, "problem_id": q, "correct": c,
                                  "tokens": {k: list(tokens[k][q]) for k in CALL_MODEL}})
            rows += pair_rows
            n = len(pair_rows)
            reports.append({
                "domain": domain, "split": split, "evaluation": evaluation, "models": [w, s], "weaker": w,
                "stronger": s, "params_b": params, "n": n,
                "accuracy_pct": {k: 100 * sum(r["correct"][k] for r in pair_rows) / n for k in SYSTEMS},
                "mean_tokens": {k: [sum(r["tokens"][k][i] for r in pair_rows) / n for i in (0, 1)] for k in CALL_MODEL},
                "cost_per_question": {wn: {k: sum(system_cost(r["tokens"], k, params, wv) for r in pair_rows) / n
                                           for k in SYSTEMS} for wn, wv in OUTPUT_WEIGHTS.items()},
                "jev": jev, "archives": {k: v.name for k, v in runs.items()}})
    return {"rows": rows, "pairs": reports, "problems": problems, "notes": notes, "sources": ar.sources}


SHORT = {"weaker_alone": "W", "stronger_alone": "S", "review_by_stronger": "S审W", "review_by_weaker": "W审S",
         "selector_primary": "选择器", "selector_secondary": "选择器2", "JEV": "JEV", "oracle": "上限"}


def write(result: dict, out: Path) -> None:
    (out / "db").mkdir(parents=True)
    (out / "reports").mkdir()
    (out / "db/cost_rows.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in result["rows"]),
                                            encoding="utf-8")
    (out / "reports/cost_pairs.json").write_text(json.dumps(
        {"unit": "billion parameters x tokens per question", "output_weights": OUTPUT_WEIGHTS,
         "jev_usd_per_million_input": JEV_USD_PER_M_INPUT, "params_b": PARAMS_B, "pairs": result["pairs"]},
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "reports/inventory.json").write_text(json.dumps(
        {"source_sha256": result["sources"],
         "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2) + "\n")
    shown = [k for k in SYSTEMS if k != "selector_secondary"]
    lines = ["# RQ3/RQ4 成本表", "",
             "成本 = 参数量（十亿）×（输入 + 输出 token），每题平均；倍数都相对较强者单独作答（S）。逐题数据在 `db/`，只留服务器。",
             "选择器和 JEV 的成本是两次单独作答；JEV 自己的 API 费用单独按美元列，不加进来。", "",
             "|域|W / S|题数|" + "|".join(f"{SHORT[k]} 准确率" for k in shown) + "|"
             + "|".join(f"{SHORT[k]} 成本倍数" for k in ("weaker_alone", "review_by_stronger", "review_by_weaker",
                                                     "selector_primary")) + "|JEV 美元/千题|",
             "|---|---|---:|" + "---:|" * (len(shown) + 5)]
    for p in result["pairs"]:
        cost = p["cost_per_question"]["main"]
        lines.append(f"|{p['domain']}|{p['weaker']} / {p['stronger']}|{p['n']}|"
                     + "|".join(f"{p['accuracy_pct'][k]:.1f}" for k in shown) + "|"
                     + "|".join(f"{cost[k] / cost['stronger_alone']:.2f}" for k in
                                ("weaker_alone", "review_by_stronger", "review_by_weaker", "selector_primary"))
                     + f"|{p['jev']['usd_per_1k_questions']:.4f}|")
    for title, key in (("没有纳入的配对（原因）", "problems"), ("提示", "notes")):
        lines += ["", f"## {title}", ""] + ([f"- {x}" for x in result[key]] or ["- 无"])
    (out / "reports/summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    if args.out_dir.exists():
        sys.exit(f"{args.out_dir} exists; completed outputs are never overwritten")
    result = build(args.db, args.artifacts)
    write(result, args.out_dir)
    print(f"{len(result['pairs'])} pairs, {len(result['rows'])} rows, {len(result['problems'])} pairs left out")


if __name__ == "__main__":
    main()
