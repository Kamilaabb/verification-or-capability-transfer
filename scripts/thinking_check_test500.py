#!/usr/bin/env python3
"""Thinking-mode sanity check: Qwen3-8B with thinking on, alone and reviewing Llama-3.1-8B (c068) and Qwen3-32B
(c097), on the first 500 test1500 math questions (docs/plan/thinking_check_test500.md). Server only.

    generate  --api-base URL --served-model NAME --records DIR [--concurrency 64]
              [--api-key-file PATH --server-label LABEL --dashscope]
    analyze   --records DIR --artifacts DIR --out DIR

`generate` writes records/solo.jsonl, records/review_c068.jsonl and records/review_c097.jsonl, one row per
question, and resumes by skipping rows already written. Prompts, answer extraction, grader and the delivery
rule (an unparseable review delivers G's answer) are those of the registered runs; only thinking and its
sampling differ (see the plan). `analyze` grades the three files and compares each review direction with its
thinking-off counterpart (c092 alone, c236 and c253 reviews) on the same questions, from the archives. Counts
only; no fit, no threshold. It writes a new directory and never overwrites.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import run_mathfull_review as review  # noqa: E402

from authority_allocation import derive_call_seed  # noqa: E402
from authority_allocation.api import api_call  # noqa: E402
from authority_allocation.prompts import render, split_prompt  # noqa: E402

N = 500
MANIFEST = ROOT / "data/manifests/mathfull-r465bcdb-l3to5-s20260924/test1500.jsonl"
SOLVE_PROMPT = "conditions/c025/prompts/g_solve_v2.txt"
SNAPSHOT = "b968826d9c46dd6066d109eabc6255188de91218"  # Qwen3-8B, as c092
DECODING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "max_new_tokens": 32768,
            "frequency_penalty": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0}
# Window 40,960 tokens: a prompt longer than 8,192 tokens leaves less than 32,768 for the output.
BUDGETS = (32768, 28672, 24576, 20480, 16384, 8192)
DASHSCOPE_ANSWER_TOKENS = 8192  # Model Studio's maximum output for qwen3-8b
SEED = ("r012", "p01", 20260925, "thinking_check_v1")
# call: (thinking-off counterpart, G solo condition or None for the reviewer alone)
CALLS = {"solo": ("c092", None), "review_c068": ("c236", "c068"), "review_c097": ("c253", "c097")}
SOLO_ARCHIVES = {"c092": "c092", "c068": "c068", "c097": "c097"}


def jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def manifest_ids() -> list[str]:
    return [str(json.loads(line)["problem_id"]) for line in MANIFEST.read_text().splitlines()][:N]


def archive(artifacts: Path, cid: str) -> Path:
    hits = sorted(p for p in artifacts.glob(f"*__{cid}__*test1500*")
                  if p.is_dir() and (p / "checksums.sha256").exists())
    if len(hits) != 1:
        raise SystemExit(f"{cid}: expected one archived test1500 run, found {hits}")
    return hits[0]


def g_outputs(cid: str) -> dict[str, str]:
    """G's raw test outputs, from the archive the registered review read (checksums verified by the sbatch)."""
    path = Path(review.resolve_condition(ROOT / "conditions" / CALLS[f"review_{cid}"][0] / "condition.yaml",
                                         repository_root=ROOT)["generator_source"]["artifact_path"])
    return {str(r["problem_id"]): r["outputs"][0]["raw_output"] or "" for r in jsonl(path / "records/pass1.jsonl")}


def dashscope_call(api_base: str, model: str, msgs: list[dict], seed: int, budget: int,
                   key: str) -> tuple[str, str | None, dict]:
    """Alibaba Model Studio (DashScope, OpenAI-compatible mode) serves open Qwen3 with thinking only as a stream.
    Thinking is capped by `thinking_budget` (the model then stops thinking and answers) and the answer by
    `max_tokens`, where vLLM caps both together; min_p and the penalties are not sent (all at their neutral
    values in DECODING)."""
    body = {"model": model, "messages": msgs, "stream": True, "stream_options": {"include_usage": True},
            "enable_thinking": True, "thinking_budget": budget, "max_tokens": DASHSCOPE_ANSWER_TOKENS,
            "temperature": DECODING["temperature"], "top_p": DECODING["top_p"], "top_k": DECODING["top_k"],
            "seed": seed}
    req = Request(api_base.rstrip("/") + "/chat/completions", data=json.dumps(body).encode("utf-8"), method="POST",
                  headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}",
                           "User-Agent": "thinking-check/1"})
    content, reasoning, finish, usage = [], [], None, {}
    started = time.perf_counter()
    with urlopen(req, timeout=7200) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                reasoning.append(delta.get("reasoning_content") or "")
                content.append(delta.get("content") or "")
                finish = choice.get("finish_reason") or finish
    details = usage.get("completion_tokens_details") or {}
    return "".join(content), "".join(reasoning) or None, {
        "input_tokens": usage.get("prompt_tokens"), "output_tokens": usage.get("completion_tokens"),
        "reasoning_tokens": details.get("reasoning_tokens"), "finish_reason": finish,
        "latency_ms": (time.perf_counter() - started) * 1000, "thinking_budget": budget,
        "max_answer_tokens": DASHSCOPE_ANSWER_TOKENS}


def generate(args: argparse.Namespace) -> None:
    review.configure("c236")  # test1500 loaders
    ids = manifest_ids()
    sources = review.base.load_source_rows(ids)
    model = review.condition()["model"]
    if model["revision"] != SNAPSHOT or model["prompt_pass1"] != "conditions/c027/prompts/v_reviewer_v2.txt":
        raise SystemExit("c236's reviewer is not the frozen Qwen3-8B snapshot and review prompt")
    review.solo.require_frozen_prompt(model["prompt_pass1"])
    system, user = split_prompt(ROOT / SOLVE_PROMPT)
    g = {cid: g_outputs(cid) for cid in ("c068", "c097")}
    args.records.mkdir(parents=True, exist_ok=True)
    # Two queued jobs (different partitions) may share these records; only one may write them at a time.
    import fcntl
    lock = (args.records / ".generate.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("another job is writing these records")
    jobs = []
    for call, (_, g_cid) in CALLS.items():
        out = args.records / f"{call}.jsonl"
        done = {r["problem_id"] for r in jsonl(out)}
        for pid in ids:
            if pid in done:
                continue
            if g_cid is None:
                msgs = [{"role": "system", "content": system},
                        {"role": "user", "content": render(user, problem=sources[pid]["problem"])}]
            else:
                msgs = review.messages(model, sources[pid]["problem"], g[g_cid][pid])
            seed = derive_call_seed(SEED[0], f"thinking_{call}", SEED[1], SEED[2], SEED[3], pid, 0)
            jobs.append((out, call, pid, msgs, seed))
    print(json.dumps({"stage": "generate", "pending": len(jobs)}), flush=True)
    key = args.api_key_file.read_text().strip() if args.api_key_file else "local"

    def call_once(msgs, seed, budget):
        """One request; a hosted API that refuses for rate limit or overload is asked again a little later."""
        for attempt in range(6):
            try:
                if args.dashscope:
                    return dashscope_call(args.api_base, args.served_model, msgs, seed, budget, key)
                return api_call(api_base=args.api_base, served_model=args.served_model, messages=msgs,
                                decoding={**DECODING, "max_new_tokens": budget}, seed=seed, thinking=True,
                                timeout_seconds=7200, api_key=key)
            except HTTPError as exc:
                if exc.code not in (429, 500, 502, 503, 504) or attempt == 5:
                    raise
                time.sleep(30 * (attempt + 1))

    def one(job):
        """One call. vLLM refuses (HTTP 400) a request whose prompt plus max_tokens exceeds the 40,960-token window,
        as a review of a long G output does with 32,768 new tokens (job 65705091), so the budget steps down until
        the request fits; the budget used is recorded."""
        out, call, pid, msgs, seed = job
        for budget in BUDGETS:
            try:
                content, reasoning, usage = call_once(msgs, seed, budget)
            except HTTPError as exc:
                if exc.code == 400 and budget != BUDGETS[-1]:
                    continue
                raise
            if reasoning is None and "</think>" in content:
                # A server without a reasoning parser returns the thinking inline; the answer is what follows it.
                reasoning, content = content.split("</think>", 1)
                reasoning, content = reasoning.replace("<think>", "", 1).strip(), content.strip()
            usage["max_new_tokens"] = budget
            return out, call, pid, seed, content, reasoning, usage

    # Answers are extracted here, in the main thread: math_verify's parsing timeout uses signals, which
    # Python allows only in the main thread (job 65704832 lost every row it parsed in a worker thread).
    done = 0
    pool = ThreadPoolExecutor(max_workers=args.concurrency)
    try:
        for fut in as_completed([pool.submit(one, j) for j in jobs]):
            out, call, pid, seed, content, reasoning, usage = fut.result()
            if usage.get("finish_reason") == "abort":
                # The server aborted the request (it was stopping); an empty output is not a model answer.
                raise RuntimeError(f"{call} {pid}: the server aborted the request")
            extracted, status = review.shared.extract_answer(content)
            row = {"problem_id": pid, "call": call, "seed": seed, "outputs": [{
                "raw_output": content, "reasoning": reasoning, "extracted_answer": extracted, "parse_status": status,
                "usage": usage}], "server": args.server_label}
            with out.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            done += 1
            if done % 50 == 0:
                print(json.dumps({"done": done, "of": len(jobs)}), flush=True)
    except BaseException:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown()
    for call in CALLS:
        rows = jsonl(args.records / f"{call}.jsonl")
        if sorted(r["problem_id"] for r in rows) != sorted(ids):
            raise SystemExit(f"{call}: {len(rows)} rows, not the first {N} manifest rows once each")
    print(json.dumps({"stage": "generate", "complete": True}), flush=True)


def answer(row: dict) -> str | None:
    a = row["outputs"][0].get("extracted_answer")
    return a.strip() if isinstance(a, str) and a.strip() else None


def analyze(args: argparse.Namespace) -> None:
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; reports are never overwritten")
    review.configure("c236")
    base = review.base
    ids = [p for p in manifest_ids() if p not in review.solo.ungradeable_ids()]
    gold = {p: r["gold"] for p, r in base.load_source_rows(ids).items()}

    def grade(row: dict, fallback: dict | None = None) -> int:
        """A thinking-on answer graded as the registered runs grade theirs; a review falls back to G's answer."""
        text = base.answer_text(row["outputs"][0])
        if text is None and fallback is not None:
            text = base.answer_text(fallback["outputs"][0])
        return int(base.grade_prediction(gold[row["problem_id"]], text)[0] or 0)

    think = {call: {r["problem_id"]: r for r in jsonl(args.records / f"{call}.jsonl")} for call in CALLS}
    aborted = [(c, p) for c, rows in think.items() for p, r in rows.items()
               if r["outputs"][0]["usage"].get("finish_reason") == "abort"]
    if aborted:
        raise SystemExit(f"{len(aborted)} rows hold server-aborted requests, not answers: {aborted[:5]}")
    servers = {call: {} for call in CALLS}
    for call, rows in think.items():
        for r in rows.values():
            label = r.get("server", "vllm-cluster")  # rows written before the label existed
            servers[call][label] = servers[call].get(label, 0) + 1
    solo_rows = {cid: {str(r["problem_id"]): r for r in jsonl(archive(args.artifacts, cid) / "records/pass1.jsonl")}
                 for cid in SOLO_ARCHIVES}
    solo_u = {cid: {str(r["problem_id"]): int(r["u"] or 0)
                    for r in jsonl(archive(args.artifacts, cid) / "records/grades.jsonl") if r["outcome"] == "u0"}
              for cid in SOLO_ARCHIVES}
    v_think_u = {p: grade(think["solo"][p]) for p in ids}
    stats = {"n": len(ids), "servers": servers, "solo": {
        "thinking_off": sum(solo_u["c092"][p] for p in ids), "thinking_on": sum(v_think_u.values()),
        "length_stops_on": sum(think["solo"][p]["outputs"][0]["usage"].get("finish_reason") == "length" for p in ids),
        "mean_output_tokens_on": sum(think["solo"][p]["outputs"][0]["usage"].get("output_tokens") or 0
                                     for p in ids) / len(ids)}, "directions": []}
    pct = lambda a, b: f"{a}/{b} ({100 * a / b:.1f}%)" if b else "n/a"
    for call, (orig, g_cid) in CALLS.items():
        if g_cid is None:
            continue
        g_rows, g_u = solo_rows[g_cid], solo_u[g_cid]
        old_dir = archive(args.artifacts, orig)
        old_rows = {str(r["problem_id"]): r for r in jsonl(old_dir / "records/review.jsonl")}
        old_u = {str(r["problem_id"]): int(r["u"] or 0) for r in jsonl(old_dir / "records/grades.jsonl")
                 if r["outcome"] == "u_gv"}
        new_rows = think[call]
        new_u = {p: grade(new_rows[p], g_rows[p]) for p in ids}
        row = {"call": call, "thinking_off_condition": orig, "g_condition": g_cid}
        for mode, v_u, v_rows, rv_u, rv_rows in (("off", solo_u["c092"], solo_rows["c092"], old_u, old_rows),
                                                 ("on", v_think_u, think["solo"], new_u, new_rows)):
            only_v = [p for p in ids if v_u[p] == 1 and g_u[p] == 0]
            only_g = [p for p in ids if g_u[p] == 1 and v_u[p] == 0]
            acc = {"g": sum(g_u[p] for p in ids), "v": sum(v_u[p] for p in ids), "review": sum(rv_u[p] for p in ids)}
            stronger = "reviewer" if acc["v"] >= acc["g"] else "G"
            same = lambda a, b: a is not None and a == b
            row[mode] = {
                "accuracy": acc, "stronger": stronger,
                "vs_stronger_pp": 100 * (acc["review"] - max(acc["v"], acc["g"])) / len(ids),
                "vs_g_pp": 100 * (acc["review"] - acc["g"]) / len(ids),
                "only_v": len(only_v), "only_v_kept": sum(rv_u[p] for p in only_v),
                "only_g": len(only_g), "only_g_kept": sum(rv_u[p] for p in only_g),
                "final_is_g_answer": sum(same(answer(rv_rows[p]), answer(g_rows[p])) for p in ids),
                "final_is_v_solo_answer": sum(same(answer(rv_rows[p]), answer(v_rows[p])) for p in ids),
                "length_stops": sum(rv_rows[p]["outputs"][0].get("usage", {}).get("finish_reason") == "length"
                                    for p in ids),
                "mean_output_tokens": sum(rv_rows[p]["outputs"][0].get("usage", {}).get("output_tokens") or 0
                                          for p in ids) / len(ids)}
        stats["directions"].append(row)

    s = stats["solo"]
    lines = ["# Thinking-mode sanity check: Qwen3-8B, first 500 test1500 math questions", "",
             f"Questions used: {stats['n']} (the first 500 minus registered ungradeable gold). Thinking on: "
             "temperature 0.6, top_p 0.95, top_k 20, up to 32,768 new tokens; thinking off: the registered runs.", "",
             f"Qwen3-8B alone: {s['thinking_off']} right with thinking off, {s['thinking_on']} with thinking on "
             f"({s['length_stops_on']} stopped at the token limit; mean {s['mean_output_tokens_on']:.0f} output tokens).", "",
             "|direction|thinking|G / reviewer alone / review|stronger|review vs stronger (pp)|review vs G (pp)|"
             "only reviewer right: kept|only G right: kept|final = G's answer|final = reviewer's solo answer|"
             "length stops|mean output tokens|",
             "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for d in stats["directions"]:
        for mode in ("off", "on"):
            m = d[mode]
            a = m["accuracy"]
            lines.append(f"|Qwen3-8B reviews {d['g_condition']} ({d['thinking_off_condition'] if mode == 'off' else 'thinking'})|"
                         f"{mode}|{a['g']} / {a['v']} / {a['review']}|{m['stronger']}|{m['vs_stronger_pp']:+.1f}|"
                         f"{m['vs_g_pp']:+.1f}|{pct(m['only_v_kept'], m['only_v'])}|{pct(m['only_g_kept'], m['only_g'])}|"
                         f"{m['final_is_g_answer']}|{m['final_is_v_solo_answer']}|{m['length_stops']}|"
                         f"{m['mean_output_tokens']:.0f}|")
    lines += ["", "G is c068 (Llama-3.1-8B-Instruct) or c097 (Qwen3-32B), thinking off in both modes; only the "
              "reviewer's mode changes. A review that gives no parseable answer delivers G's answer.", "",
              "Thinking-on rows by server: " + "; ".join(
                  f"{call} " + ", ".join(f"{k} {v}" for k, v in sorted(c.items())) for call, c in servers.items())]
    args.out.mkdir(parents=True)
    (args.out / "reports").mkdir()
    (args.out / "reports/result.json").write_text(json.dumps(stats, indent=1) + "\n")
    (args.out / "reports/summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--api-base", required=True)
    g.add_argument("--served-model", required=True)
    g.add_argument("--records", type=Path, required=True)
    g.add_argument("--concurrency", type=int, default=64)
    g.add_argument("--api-key-file", type=Path, help="file holding the API key of a hosted server")
    g.add_argument("--server-label", default="vllm-cluster", help="who served the calls, written on each row")
    g.add_argument("--dashscope", action="store_true", help="the server is Alibaba Model Studio (streamed thinking)")
    a = sub.add_parser("analyze")
    a.add_argument("--records", type=Path, required=True)
    a.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    a.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    (generate if args.cmd == "generate" else analyze)(args)


if __name__ == "__main__":
    main()
