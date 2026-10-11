#!/usr/bin/env python3
"""JEV as an external judge between G and V on MMLU-Pro test1500 (docs/plan/mmlu_pro_study.md section 17).

Analysis only; the rules were fixed in section 17 before any call. For each pair (V = the model with the
higher dev1000 solo accuracy), every test row where G's and V's registered answer letters differ is sent to
TypeSafe's JEV (`jev-latest`) twice, once per order, as a two-option choice: "Response A's final answer is
correct" / "Response B's...". The state holds the question, the options and the two full solo outputs; no
gold answer, grade or label is ever sent. P(V) = 1/2 [p(V shown as B) + p(V shown as A)]; the direct
decision delivers V when P(V) >= 0.5 (a tie delivers V, as the selectors do), else G. Rows where the letters
agree deliver that letter without a call. A row whose call still fails after retries delivers V and is
counted. Every raw response is kept; `/v1/models` is recorded once at the start.

Runs on the cluster login node (the only place with network access); the API key is read from
~/.typesafe_key and never printed. Responses are cached per (pair, row, order), so a rerun resumes.

  python scripts/mc_jev_judge_test1500.py --smoke 20 --out /scratch/.../jev_smoke
  python scripts/mc_jev_judge_test1500.py --out /scratch/.../<stem>   # all pairs, then archive
  python scripts/mc_jev_judge_test1500.py --out <same dir> --latency 100   # timing sample, see latency()
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from authority_allocation import mmlu_pro  # noqa: E402
from test_mathfull_v2_analysis import sign_p  # noqa: E402

API = "https://api.typesafe.ai/v1"
MODEL = "jev-latest"
KEY_FILE = Path("~/.typesafe_key").expanduser()
ART = ROOT / "artifacts"
TEST_MANIFEST = ROOT / "data/manifests/mmlupro-rb189ec7-s20260927/test1500.jsonl"
INSTRUCTIONS = "Which response gives the correct final answer to the multiple-choice question?"
CRITERIA = {"A": "Response A's final answer is correct", "B": "Response B's final answer is correct"}

# Stratified pairs without a Qwen2.5-72B substitution (sections 15-16), all models confirmed on test1500.
# name, (G model, G test solo), (V model, V test solo), [(GV condition, note)]
PAIRS = [
    ("qwen25-7b__qwen3-4b", ("Qwen2.5-7B", "c122"), ("Qwen3-4B", "c081"),
     [("c136", "Qwen2.5-7B reviews Qwen3-4B"), ("c137", "Qwen3-4B reviews Qwen2.5-7B")]),
    ("qwen25-7b__qwen3-8b", ("Qwen2.5-7B", "c122"), ("Qwen3-8B", "c082"),
     [("c138", "Qwen2.5-7B reviews Qwen3-8B"), ("c139", "Qwen3-8B reviews Qwen2.5-7B")]),
    ("ministral-8b__qwen25-3b", ("Ministral-8B", "c123"), ("Qwen2.5-3B", "c121"),
     [("c140", "Ministral-8B reviews Qwen2.5-3B"), ("c141", "Qwen2.5-3B reviews Ministral-8B")]),
    ("llama-32-3b__qwen25-3b", ("Llama-3.2-3B", "c120"), ("Qwen2.5-3B", "c121"),
     [("c142", "Llama-3.2-3B reviews Qwen2.5-3B"), ("c143", "Qwen2.5-3B reviews Llama-3.2-3B")]),
    ("qwen25-3b__qwen25-7b", ("Qwen2.5-3B", "c121"), ("Qwen2.5-7B", "c122"),
     [("c144", "Qwen2.5-3B reviews Qwen2.5-7B"), ("c145", "Qwen2.5-7B reviews Qwen2.5-3B")]),
    ("llama-32-3b__qwen3-8b", ("Llama-3.2-3B", "c120"), ("Qwen3-8B", "c082"),
     [("c146", "Llama-3.2-3B reviews Qwen3-8B"), ("c147", "Qwen3-8B reviews Llama-3.2-3B")]),
    ("qwen3-8b__mistral-24b", ("Qwen3-8B", "c082"), ("Mistral-Small-24B", "c089"),
     [("c176", "Mistral-Small-24B reviews Qwen3-8B"), ("c177", "Qwen3-8B reviews Mistral-Small-24B")]),
]


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def archive(cid: str) -> Path | None:
    hits = sorted(ART.glob(f"*__{cid}__mmlupro-test1500__*"))
    return hits[-1] if hits and (hits[-1] / "checksums.sha256").exists() else None


def solo(cid: str) -> dict[str, dict]:
    """problem_id -> {letter, text, u} from a completed test1500 solo archive."""
    arch = archive(cid)
    if arch is None:
        raise SystemExit(f"{cid}: no completed test1500 archive")
    u = {r["problem_id"]: int(r["u"]) for r in read_jsonl(arch / "records/grades.jsonl") if r["outcome"] == "u0"}
    out = {}
    for r in read_jsonl(arch / "records/pass1.jsonl"):
        o = r["outputs"][0]
        out[r["problem_id"]] = {"letter": o.get("extracted_answer"), "text": o.get("raw_output") or "",
                                "u": u[r["problem_id"]]}
    if len(out) != 1500 or len(u) != 1500:
        raise SystemExit(f"{cid}: expected 1500 rows")
    return out


def gv(cid: str) -> dict[str, int] | None:
    arch = archive(cid)
    if arch is None:
        return None
    return {r["problem_id"]: int(r["u"]) for r in read_jsonl(arch / "records/grades.jsonl") if r["outcome"] == "u_gv"}


def load_questions(problem_ids: list[str]) -> dict[str, dict]:
    """Question and options only (never the answer) from the pinned source, as run_mmlu_pro does."""
    from datasets import load_dataset

    wanted, rows = set(problem_ids), {}
    for row in load_dataset("TIGER-Lab/MMLU-Pro", revision="b189ec765aa7ed75c8acfea42df31fdae71f97be", split="test"):
        pid = str(row["question_id"])
        if pid in wanted:
            rows[pid] = {"question": row["question"], "options": list(row["options"])}
    if len(rows) != len(wanted):
        raise SystemExit("manifest ids missing from the pinned source")
    return rows


def state(question: str, options: list[str], a: str, b: str) -> str:
    return (f"Question:\n{question}\n\nOptions:\n{mmlu_pro.render_options(options)}\n\n"
            f"Response A:\n{a}\n\nResponse B:\n{b}")


def request(path: str, key: str, body: dict | None = None, tries: int = 6) -> dict:
    data = None if body is None else json.dumps(body).encode()
    for attempt in range(tries):
        req = urllib.request.Request(f"{API}/{path}", data=data, method="GET" if body is None else "POST",
                                     headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code in (401, 422) or attempt == tries - 1:
                raise RuntimeError(f"HTTP {e.code}: {e.read()[:300]!r}") from None
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == tries - 1:
                raise RuntimeError(f"network: {e}") from None
        time.sleep(min(60, 2 ** attempt * 2))
    raise RuntimeError("unreachable")


def judge(key: str, st: str) -> dict:
    body = {"model": MODEL, "state": st,
            "questions": {"correct": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": CRITERIA}}}
    return request("systemone", key, body)


def quantile(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    i = (len(xs) - 1) * q
    lo, hi = int(i), min(int(i) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def latency(args, key: str, rows_by_pair: dict, source: dict) -> int:
    """Wall-clock latency of single JEV calls: one request at a time, no retry, from the login node.
    The rows are every k-th disagreement row of each pair (deterministic), both orders. The decisions
    are not used; this only measures time."""
    per_pair = max(1, args.latency // len(rows_by_pair))
    items = []
    for name, (g, v, dis) in rows_by_pair.items():
        step = max(1, len(dis) // per_pair)
        for pid in dis[::step][:per_pair]:
            q = source[pid]
            for order, (a, b) in (("VG", (v[pid]["text"], g[pid]["text"])), ("GV", (g[pid]["text"], v[pid]["text"]))):
                items.append((name, pid, order, state(q["question"], q["options"], a, b)))
    out = args.out / "records/jev_latency.jsonl"
    recs = []
    with out.open("a", encoding="utf-8") as f:
        for name, pid, order, st in items:
            body = {"model": MODEL, "state": st,
                    "questions": {"correct": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": CRITERIA}}}
            req = urllib.request.Request(f"{API}/systemone", data=json.dumps(body).encode(), method="POST",
                                         headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            rec = {"pair": name, "problem_id": pid, "order": order,
                   "called_at_utc": datetime.now(timezone.utc).isoformat()}
            t0 = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    r = json.loads(resp.read())
                rec["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                rec["model"] = r.get("model")
                rec["input_tokens"] = int(r.get("usage", {}).get("input_tokens", 0))
            except (urllib.error.URLError, TimeoutError) as e:
                rec["error"] = str(e)
            f.write(json.dumps(rec) + "\n")
            recs.append(rec)
    ok = [r for r in recs if "error" not in r]
    lat = [r["latency_ms"] for r in ok]
    lines = ["# JEV call latency (single calls, one at a time, from the cluster login node)", "",
             f"{len(ok)} successful calls of {len(recs)}; model {sorted({r['model'] for r in ok})}.", "",
             "|statistic|ms|", "|---|---|"]
    for label, q in (("median", .5), ("p10", .1), ("p90", .9), ("p99", .99)):
        lines.append(f"|{label}|{quantile(lat, q):.0f}|")
    lines.append(f"|mean|{sum(lat) / len(lat):.0f}|")
    lines.append(f"|max|{max(lat):.0f}|")
    lines += ["", "|input tokens|calls|median ms|", "|---|---|---|"]
    for lo, hi in ((0, 1000), (1000, 2000), (2000, 4000), (4000, 10 ** 9)):
        sel = [r["latency_ms"] for r in ok if lo <= r["input_tokens"] < hi]
        if sel:
            lines.append(f"|{lo}-{hi if hi < 10 ** 9 else ''}|{len(sel)}|{quantile(sel, .5):.0f}|")
    (args.out / "reports/latency.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True, help="working directory (cache, reports)")
    ap.add_argument("--smoke", type=int, default=0, help="only the first pair's first N disagreement rows")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--latency", type=int, default=0,
                    help="timing sample only: N disagreement rows spread over the pairs, both orders, one call at a time")
    args = ap.parse_args()
    key = KEY_FILE.read_text().strip()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "records").mkdir(exist_ok=True)
    (args.out / "reports").mkdir(exist_ok=True)

    models = request("models", key)
    with (args.out / "records/jev_models.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"queried_at_utc": datetime.now(timezone.utc).isoformat(), "response": models}) + "\n")

    manifest = [str(r["problem_id"]) for r in read_jsonl(TEST_MANIFEST)]
    source = load_questions(manifest)

    cache_path = args.out / "records/jev_responses.jsonl"
    cache = {(r["pair"], r["problem_id"], r["order"]): r for r in (read_jsonl(cache_path) if cache_path.exists() else [])
             if "error" not in r}
    pairs = PAIRS[:1] if args.smoke else PAIRS
    todo, rows_by_pair = [], {}
    for name, (gname, gcid), (vname, vcid), _ in pairs:
        g, v = solo(gcid), solo(vcid)
        dis = [pid for pid in manifest if g[pid]["letter"] != v[pid]["letter"]]
        if args.smoke:
            dis = dis[:args.smoke]
        rows_by_pair[name] = (g, v, dis)
        for pid in dis:
            q = source[pid]
            for order, (a, b) in (("VG", (v[pid]["text"], g[pid]["text"])), ("GV", (g[pid]["text"], v[pid]["text"]))):
                if (name, pid, order) not in cache:
                    todo.append((name, pid, order, state(q["question"], q["options"], a, b)))

    def call(item):
        name, pid, order, st = item
        rec = {"pair": name, "problem_id": pid, "order": order, "model_requested": MODEL,
               "state_sha256": hashlib.sha256(st.encode()).hexdigest(),
               "called_at_utc": datetime.now(timezone.utc).isoformat()}
        t0 = time.perf_counter()
        try:
            rec["response"] = judge(key, st)
        except RuntimeError as e:
            rec["error"] = str(e)
        rec["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)  # includes any retries
        return rec

    if args.latency:
        return latency(args, key, rows_by_pair, source)
    print(f"{len(todo)} calls to make ({len(cache)} cached)", flush=True)
    with cache_path.open("a", encoding="utf-8") as f, cf.ThreadPoolExecutor(args.workers) as pool:
        for i, rec in enumerate(pool.map(call, todo), 1):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if "error" not in rec:
                cache[(rec["pair"], rec["problem_id"], rec["order"])] = rec
            if i % 100 == 0:
                print(f"{i}/{len(todo)}", flush=True)

    # ---- decisions and report
    lines = ["# JEV as an external judge between G and V on MMLU-Pro test1500", "",
             "Analysis only (docs/plan/mmlu_pro_study.md section 17). V = the model with the higher dev1000 solo "
             "accuracy, so max(G, V) = V on dev. JEV (`jev-latest`) judges only rows where G's and V's letters "
             "differ, once per order; P(V) averages the two orders; P(V) >= 0.5 delivers V. No gold answer or "
             "grade was sent. GV = the registered primary delivery of the test GV run (where archived).", "",
             "|pair|G|V|max(G,V)|oracle|disagree rows|JEV|JEV - max|JEV vs max (JEV:max, p)|JEV picks V|failed|"
             "input tokens|GV runs|", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    summary, tokens_total = [], 0
    for name, (gname, gcid), (vname, vcid), gvs in pairs:
        g, v, dis = rows_by_pair[name]
        ids = manifest if not args.smoke else dis
        ug = [g[p]["u"] for p in ids]
        uv = [v[p]["u"] for p in ids]
        pick_v, failed, tok = {}, 0, 0
        for pid in dis:
            ps = []
            for order in ("VG", "GV"):
                rec = cache.get((name, pid, order))
                if rec is None:
                    continue
                probs = rec["response"]["answers"]["correct"]["probabilities"]
                tok += int(rec["response"].get("usage", {}).get("input_tokens", 0))
                ps.append(float(probs["A"]) if order == "VG" else float(probs["B"]))
            if len(ps) < 2:
                failed += 1
                pick_v[pid] = True
            else:
                pick_v[pid] = sum(ps) / 2 >= 0.5
        uj = [(v[p]["u"] if pick_v.get(p, True) else g[p]["u"]) for p in ids]
        mx = uv if sum(uv) >= sum(ug) else ug
        wins = sum(1 for a, b in zip(uj, mx) if a > b)
        losses = sum(1 for a, b in zip(uj, mx) if a < b)
        pval = sign_p(wins, losses)
        gv_txt = []
        for cid, note in gvs:
            ugv = gv(cid)
            gv_txt.append(f"{cid} {sum(ugv[x] for x in ids)}" if ugv else f"{cid} pending")
        oracle = sum(max(a, b) for a, b in zip(ug, uv))
        tokens_total += tok
        lines.append(f"|{name}|{gname} {sum(ug)}|{vname} {sum(uv)}|{sum(mx)}|{oracle}|{len(dis)}|{sum(uj)}|"
                     f"{sum(uj) - sum(mx):+d}|{wins}:{losses}, p={pval:.3f}|{sum(pick_v.values())}|{failed}|{tok}|"
                     f"{'; '.join(gv_txt)}|")
        summary.append({"pair": name, "g": [gname, gcid, sum(ug)], "v": [vname, vcid, sum(uv)], "max": sum(mx),
                        "oracle": oracle, "disagree_rows": len(dis), "jev": sum(uj), "jev_vs_max": [wins, losses, pval],
                        "jev_picks_v": sum(pick_v.values()), "failed_rows": failed, "input_tokens": tok,
                        "gv": gv_txt})
    lines += ["", f"Input tokens in total: {tokens_total} (about ${tokens_total * 0.042 / 1e6:.2f} at $0.042 per million).",
              f"`/v1/models` at the start: see records/jev_models.jsonl."]
    if args.smoke:
        lines.insert(2, f"**Smoke test: first pair, first {args.smoke} disagreement rows only.**")
    (args.out / "reports/summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.out / "reports/jev_judge.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
