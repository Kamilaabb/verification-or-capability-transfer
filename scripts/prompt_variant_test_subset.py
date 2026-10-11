#!/usr/bin/env python3
"""Commit-first review prompt on test1500, first 500 questions (c390-c393; docs/plan/prompt_variant_test500.md).
Server only.

    python scripts/prompt_variant_test_subset.py --db /path/to/scratch/results_db/20261007 \
        --runs /path/to/scratch/authority-allocation/runs --artifacts artifacts \
        --self-review docs/analysis/claims/20261009_self_review_test500/result.json \
        --out /path/to/scratch/claims/<new dir>

The four runs were stopped after row 500 and are not archived. For each, this reads the first 500 review
rows from the run directory, checks they are the first 500 manifest rows in order, and grades them with
the condition's own runner (`grade` stage, into the output directory). The original-prompt review of the
same direction comes from the results database (grades) and its archive (extracted answers); the
reviewer's solo answers come from its solo archive.

Per direction, on the same 500 questions, for the original prompt and the commit-first prompt: accuracy
of G alone, the reviewer alone and each review, and the change over the stronger model alone; on the
questions only the reviewer gets right alone, and on those only G gets right alone, the share still right
after review; how often the final answer equals G's extracted answer and the reviewer's solo answer; the
share of review outputs that stop at the token limit. For the commit-first prompt, the committed answer
(the last "My answer:" line, read with the same extraction and grader as a final answer): how often it is
present, its accuracy, and on the questions where it is right and G's answer is wrong, how often the final
answer is still right and how often it is G's answer. Counts only; no fit, no threshold. Writes a new
directory and never overwrites.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
N = 500
MATH_MANIFEST = "data/manifests/mathfull-r465bcdb-l3to5-s20260924/test1500.jsonl"
MC_MANIFEST = "data/manifests/mmlupro-rb189ec7-s20260927/test1500.jsonl"
VARIANT = {  # condition: (domain, G, reviewer as in the results database, original-prompt condition, runner, manifest)
    "c390": ("math", "Llama-3.1-8B-Instruct", "Qwen3-8B", "c236", "run_mathfull_review.py", MATH_MANIFEST),
    "c391": ("math", "Qwen3-32B", "Qwen3-8B", "c253", "run_mathfull_review.py", MATH_MANIFEST),
    "c392": ("knowledge", "Llama-3.2-3B-Instruct", "Qwen3-8B", "c147", "run_mmlu_pro.py", MC_MANIFEST),
    "c393": ("knowledge", "Qwen3-4B", "Qwen2.5-7B-Instruct", "c136", "run_mmlu_pro.py", MC_MANIFEST),
}
MY_ANSWER = re.compile(r"^[\s>#*_-]*My answer\s*[:：]\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


def jsonl(path: Path) -> list[dict]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:  # a row cut by the time limit
            break
    return out


def answer(row: dict):
    a = row["outputs"][0].get("extracted_answer")
    return a.strip() if isinstance(a, str) and a.strip() else None


def archive(artifacts: Path, cid: str, split: str = "test1500") -> Path:
    hits = sorted(p for p in artifacts.glob(f"*__{cid}__*{split}*")
                  if p.is_dir() and (p / "checksums.sha256").exists())
    if len(hits) != 1:
        raise SystemExit(f"{cid}: expected one archived run, found {hits}")
    return hits[0]


def committed_text(raw: str | None) -> str | None:
    """The text after the last "My answer:" line, without Markdown emphasis, $ delimiters and a final period."""
    hits = MY_ANSWER.findall(raw or "")
    if not hits:
        return None
    text = hits[-1].strip().strip("*").strip().rstrip(".").strip()
    text = re.sub(r"^\$+|\$+$", "", text).strip()
    return text.rstrip(".").strip() or None


class MathCommitted:
    def __init__(self, ids, cid):
        import run_mathfull_review as runner  # configure() points the shared loaders at the condition's dataset
        runner.configure(cid)
        import run_c002_dev30 as base
        import run_c014_router as shared
        self.base, self.shared = base, shared
        self.gold = {p: r["gold"] for p, r in base.load_source_rows(ids).items()}

    def read(self, pid, text):
        """(extracted answer, right) of a committed answer, graded as if it were boxed."""
        if text is None:
            return None, 0
        boxed = "\\boxed{" + text + "}"
        extracted, _ = self.shared.extract_answer(boxed)
        u, _ = self.base.grade_prediction(self.gold[pid], boxed)
        return (extracted.strip() if isinstance(extracted, str) and extracted.strip() else None), int(u or 0)


class McCommitted:
    def __init__(self, ids, cid):
        import run_mmlu_pro
        from authority_allocation import mmlu_pro
        self.parse = mmlu_pro.parse_answer_letter
        rows = run_mmlu_pro.load_mmlu_pro_rows(ids)
        self.gold = {p: r["answer"] for p, r in rows.items()}
        self.n_options = {p: len(r["options"]) for p, r in rows.items()}

    def read(self, pid, text):
        """(letter, right) of a committed answer, read as if it were the "Answer:" line."""
        if text is None:
            return None, 0
        letter = self.parse("Answer: " + text, self.n_options[pid])
        return letter, int(letter is not None and letter == self.gold[pid])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    ap.add_argument("--self-review", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; reports are never overwritten")
    (args.out / "graded").mkdir(parents=True)

    dirs = {}
    for line in (args.db / "db/directions.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["split"] == "test" and r["condition"] in {v[3] for v in VARIANT.values()}:
            dirs.setdefault(r["condition"], {})[str(r["problem_id"])] = r
    pairs = json.loads((args.db / "db/pairs.json").read_text())
    selfs = {(r["domain"], r["reviewer"], r["g"]): r
             for r in json.loads(args.self_review.read_text())["rows"]}

    results, lines = [], []
    pct = lambda a, b: f"{a}/{b} ({100 * a / b:.1f}%)" if b else "n/a"
    for cid, (domain, g_model, reviewer, orig, runner, manifest) in VARIANT.items():
        runs = [p for p in args.runs.glob(f"*__{cid}__*") if p.is_dir() and not p.name.endswith("__shards")]
        if len(runs) != 1:
            raise SystemExit(f"{cid}: expected one run directory, found {runs}")
        run = runs[0]
        ids = [str(json.loads(l)["problem_id"]) for l in (ROOT / manifest).read_text().splitlines()][:N]
        rows = jsonl(run / "records/review.jsonl")[:N]
        if [str(r["problem_id"]) for r in rows] != ids:
            raise SystemExit(f"{cid}: the first {N} review rows are not the first {N} manifest rows")
        g_rows = {str(r["problem_id"]): r for r in jsonl(run / "records/g_pass1.jsonl")}
        sub = args.out / "graded" / cid
        sub.mkdir()
        (sub / "review.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        (sub / "g_pass1.jsonl").write_text("".join(json.dumps(g_rows[p]) + "\n" for p in ids), encoding="utf-8")
        cmd = [sys.executable, str(ROOT / "scripts" / runner), "--condition-id", cid, "grade",
               "--g-records", str(sub / "g_pass1.jsonl"), "--output", str(sub / "grades.jsonl")]
        cmd += (["--review-records", str(sub / "review.jsonl")] if domain == "math"
                else ["--records", str(sub / "review.jsonl")])
        subprocess.run(cmd, check=True, cwd=ROOT)
        new = {str(g["problem_id"]): g["u"] for g in jsonl(sub / "grades.jsonl") if g["outcome"] == "u_gv"}

        d = dirs[orig]
        missing = [i for i in ids if i not in d]
        if missing:
            raise SystemExit(f"{orig}: {len(missing)} of the first {N} test rows missing from the database")
        if {d[i]["generator"] for i in ids} != {g_model} or {d[i]["reviewer"] for i in ids} != {reviewer}:
            raise SystemExit(f"{orig}: generator or reviewer differs from {cid}")
        v_cond = {d[i]["v_condition"] for i in ids}
        if len(v_cond) != 1:
            raise SystemExit(f"{orig}: several reviewer solo runs {v_cond}")
        v_cond = v_cond.pop()
        old_rows = {str(r["problem_id"]): r for r in jsonl(archive(args.artifacts, orig) / "records/review.jsonl")}
        old_g = {str(r["problem_id"]): r for r in jsonl(archive(args.artifacts, orig) / "records/g_pass1.jsonl")}
        v_rows = {str(r["problem_id"]): r for r in jsonl(archive(args.artifacts, v_cond) / "records/pass1.jsonl")}
        # The commit-first run must have reviewed the same G outputs, byte for byte.
        changed = [p for p in ids if g_rows[p]["outputs"][0]["raw_output"] != old_g[p]["outputs"][0]["raw_output"]]
        if changed:
            raise SystemExit(f"{cid}: {len(changed)} G outputs differ from {orig}")

        keys = [i for i in ids if new[i] is not None and None not in (d[i]["g"], d[i]["v"], d[i]["gv"])]
        pair = next(p for p in pairs[domain] if p["in_design_sample"] and set(p["models"]) == {g_model, reviewer})
        kind = "stronger reviews" if pair["stronger"] == reviewer else "weaker reviews"
        reader = (MathCommitted if domain == "math" else McCommitted)(keys, cid)
        g_ans = {k: answer(g_rows[k]) for k in keys}
        v_ans = {k: answer(v_rows[k]) for k in keys}
        old_ans = {k: answer(old_rows[k]) for k in keys}
        new_ans = {k: answer(r) for k, r in zip(ids, rows) if k in g_ans}
        new_raw = {str(r["problem_id"]): r for r in rows}
        committed = {k: reader.read(k, committed_text(new_raw[k]["outputs"][0]["raw_output"])) for k in keys}
        has_committed = [k for k in keys if committed[k][0] is not None]

        only_v = [k for k in keys if d[k]["v"] == 1 and d[k]["g"] == 0]
        only_g = [k for k in keys if d[k]["g"] == 1 and d[k]["v"] == 0]
        acc = {"g": sum(d[k]["g"] for k in keys), "v": sum(d[k]["v"] for k in keys),
               "original": sum(d[k]["gv"] for k in keys), "commit_first": sum(new[k] for k in keys)}
        stronger = acc["v"] if kind == "stronger reviews" else acc["g"]
        c_right_g_wrong = [k for k in has_committed if committed[k][1] == 1 and d[k]["g"] == 0]
        same = lambda a, b: a is not None and a == b
        sr = selfs.get((domain, reviewer, g_model))
        r = {
            "condition": cid, "original_condition": orig, "domain": domain, "g": g_model, "reviewer": reviewer,
            "kind": kind, "v_condition": v_cond, "run": run.name, "n": len(keys), "accuracy": acc,
            "vs_stronger_pp": {k: 100 * (acc[k] - stronger) / len(keys) for k in ("original", "commit_first")},
            "only_v": len(only_v),
            "only_v_kept": {"original": sum(d[k]["gv"] for k in only_v), "commit_first": sum(new[k] for k in only_v)},
            "only_g": len(only_g),
            "only_g_kept": {"original": sum(d[k]["gv"] for k in only_g), "commit_first": sum(new[k] for k in only_g)},
            "final_is_g_answer": {"original": sum(same(old_ans[k], g_ans[k]) for k in keys),
                                  "commit_first": sum(same(new_ans[k], g_ans[k]) for k in keys)},
            "final_is_v_solo_answer": {"original": sum(same(old_ans[k], v_ans[k]) for k in keys),
                                       "commit_first": sum(same(new_ans[k], v_ans[k]) for k in keys)},
            "hit_token_limit": {
                "original": sum(old_rows[k]["outputs"][0].get("usage", {}).get("finish_reason") == "length" for k in keys),
                "commit_first": sum(new_raw[k]["outputs"][0].get("usage", {}).get("finish_reason") == "length"
                                    for k in keys)},
            "committed_present": len(has_committed),
            "committed_right": sum(committed[k][1] for k in has_committed),
            "committed_equals_v_solo": sum(same(committed[k][0], v_ans[k]) for k in has_committed),
            "committed_right_g_wrong": len(c_right_g_wrong),
            "of_those_final_right": sum(new[k] for k in c_right_g_wrong),
            "of_those_final_is_g_answer": sum(same(new_ans[k], g_ans[k]) for k in c_right_g_wrong),
            "self_review_only_v_kept": None if sr is None else [sr["only_v_kept_self"], sr["only_v"]],
            "hardware": (run / "logs/execution_hardware.txt").read_text().strip().splitlines(),
        }
        results.append(r)

    lines = ["# Commit-first review prompt on test1500, first 500 questions (c390-c393)", "",
             "Same directions, questions and G outputs as the original review; only the review prompt differs "
             "(the reviewer commits to its own answer first, and is no longer told to retain a correct solution).", "",
             "|direction|kind|only reviewer right: kept, original|kept, commit-first|kept, self-review|"
             "only G right: kept, original|kept, commit-first|vs stronger (pp), original|commit-first|",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        sr = r["self_review_only_v_kept"]
        lines.append(f"|{r['domain']} {r['reviewer']} reviews {r['g']} ({r['original_condition']} / {r['condition']})|"
                     f"{r['kind'].split()[0]}|{pct(r['only_v_kept']['original'], r['only_v'])}|"
                     f"{pct(r['only_v_kept']['commit_first'], r['only_v'])}|{pct(*sr) if sr else 'n/a'}|"
                     f"{pct(r['only_g_kept']['original'], r['only_g'])}|{pct(r['only_g_kept']['commit_first'], r['only_g'])}|"
                     f"{r['vs_stronger_pp']['original']:+.1f}|{r['vs_stronger_pp']['commit_first']:+.1f}|")
    lines += ["", "Accuracy of the questions used (G alone / reviewer alone / original review / commit-first review), "
              "final answer equal to G's extracted answer and to the reviewer's solo answer, outputs at the token limit:", ""]
    for r in results:
        a, f, s, t = r["accuracy"], r["final_is_g_answer"], r["final_is_v_solo_answer"], r["hit_token_limit"]
        lines.append(f"- {r['condition']}: {a['g']} / {a['v']} / {a['original']} / {a['commit_first']} of {r['n']}; "
                     f"final = G's answer {f['original']} -> {f['commit_first']}; final = reviewer's solo answer "
                     f"{s['original']} -> {s['commit_first']}; at the token limit {t['original']} -> {t['commit_first']}")
    lines += ["", "Committed answer (commit-first only):", ""]
    for r in results:
        lines.append(f"- {r['condition']}: present {r['committed_present']} of {r['n']}, right {r['committed_right']}, "
                     f"equal to the reviewer's solo answer {r['committed_equals_v_solo']}; committed right and G wrong "
                     f"{r['committed_right_g_wrong']}, of which the final answer is right "
                     f"{pct(r['of_those_final_right'], r['committed_right_g_wrong'])} and is G's answer "
                     f"{pct(r['of_those_final_is_g_answer'], r['committed_right_g_wrong'])}")
    lines += ["", "Hardware (the original reviews ran on H100):", ""] + [f"- {r['condition']}: {r['hardware']}" for r in results]
    (args.out / "reports").mkdir()
    (args.out / "reports/result.json").write_text(json.dumps({"rows": results}, indent=1) + "\n")
    (args.out / "reports/summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
