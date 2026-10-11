"""End-to-end check of build_cost_rows.py and evaluate_cost_claims.py on fake archives."""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from authority_allocation.cost_model import PARAMS_B, system_cost, usage_of  # noqa: E402

W, S = "Qwen3-4B", "Qwen3-8B"


def jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def archive(root: Path, cid: str, model: str, records: str, usage: dict, source: Path | None = None) -> Path:
    run = root / f"2026T__{cid}__x__r"
    cfg = {"condition_id": cid, "model": {"model_id": f"org/{model}"}}
    if source is not None:
        cfg["generator_source"] = {"artifact_path": f"/server/artifacts/{source.name}"}
    (run / "records").mkdir(parents=True)
    (run / "condition.resolved.yaml").write_text(yaml.safe_dump(cfg))
    jsonl(run / records, [{"problem_id": q, "outputs": [{"usage": {"input_tokens": i, "output_tokens": o}}]}
                          for q, (i, o) in usage.items()])
    (run / "checksums.sha256").write_text("".join(
        f"{hashlib.sha256((run / rel).read_bytes()).hexdigest()}  {rel}\n"
        for rel in ("condition.resolved.yaml", records)))
    return run


def make_world(tmp: Path, drop_usage: bool = False) -> tuple[Path, Path]:
    rng = random.Random(0)
    art, db = tmp / "artifacts", tmp / "db"
    solo, dirs, systems, summaries, pairs = [], [], [], [], {}
    for domain, split, n, pw, ps in (("math", "test", 120, .5, .7), ("code", "dev", 80, .3, .5)):
        ids = [f"{domain}{i}" for i in range(n)]
        yw = {q: int(rng.random() < pw) for q in ids}
        ys = {q: int(rng.random() < ps) for q in ids}
        solo_tok = {q: (100, 200) for q in ids}
        rw = archive(art, f"{domain}w", W, "records/pass1.jsonl", solo_tok)
        rs = archive(art, f"{domain}s", S, "records/pass1.jsonl", solo_tok)
        review_tok = {q: (300, 50) for q in ids}
        if drop_usage and domain == "code":
            review_tok.pop(ids[0])
        archive(art, f"{domain}sw", S, "records/review.jsonl", review_tok, source=rw)
        archive(art, f"{domain}ws", W, "records/review.jsonl", review_tok, source=rs)
        solo += [{"domain": domain, "split": split, "model": m, "problem_id": q, "correct": y[q], "condition": c}
                 for m, y, c in ((W, yw, f"{domain}w"), (S, ys, f"{domain}s")) for q in ids]
        dirs += [{"domain": domain, "split": split, "generator": g, "reviewer": v, "problem_id": q,
                  "gv": int(rng.random() < p), "condition": c}
                 for g, v, p, c in ((W, S, .65, f"{domain}sw"), (S, W, .68, f"{domain}ws")) for q in ids]
        pair_id = "|".join(sorted((W, S)))
        jev_dir = tmp / "jev" / domain
        jsonl(jev_dir / "pair/records/jev_responses.jsonl",
              [{"problem_id": q, "order": o, "response": {"usage": {"input_tokens": 1000}}}
               for q in ids[:10] for o in ("GV", "VG")])
        for name in ("selector_primary", "selector_secondary", "JEV"):
            systems += [{"domain": domain, "split": split, "evaluation": "test" if split == "test" else "cv",
                         "system": name, "pair_id": pair_id, "problem_id": q, "correct": max(yw[q], ys[q])
                         if rng.random() < .5 else ys[q]} for q in ids]
            summaries.append({"domain": domain, "models": sorted((W, S)), "system": name,
                              "upstream_runs": [rw.name, rs.name], "source": str(jev_dir)})
        pairs[domain] = [{"models": [W, S], "stronger": S, "in_design_sample": True}]
    jsonl(db / "db/solo.jsonl", solo)
    jsonl(db / "db/directions.jsonl", dirs)
    jsonl(db / "db/systems.jsonl", systems)
    (db / "db/pairs.json").write_text(json.dumps(pairs))
    (db / "reports").mkdir(parents=True)
    (db / "reports/systems.json").write_text(json.dumps({"systems": summaries}))
    return db, art


def cost_card(card_path: Path, tmp: Path) -> Path:
    card = yaml.safe_load(card_path.read_text(encoding="utf-8"))
    card["bootstrap"] = {"reps": 50, "seed": 1}
    out = tmp / card_path.name
    out.write_text(yaml.safe_dump(card, allow_unicode=True), encoding="utf-8")
    return out


class CostClaims(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def run_script(self, *args):
        run = subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        return run

    def test_usage_and_cost(self):
        self.assertEqual(usage_of({"outputs": [{"usage": {"input_tokens": 3, "output_tokens": 4}}]}), (3, 4))
        with self.assertRaises(ValueError):
            usage_of({"problem_id": "q", "outputs": [{"usage": {"input_tokens": 3}}]})
        row = {"solo_w": [100, 200], "solo_s": [100, 200], "g_of_review_by_s": [100, 200], "review_by_s": [300, 50],
               "g_of_review_by_w": [100, 200], "review_by_w": [300, 50]}
        params = {"w": 4.0, "s": 8.0}
        self.assertEqual(system_cost(row, "stronger_alone", params), 8 * 300)
        self.assertEqual(system_cost(row, "review_by_stronger", params), 4 * 300 + 8 * 350)
        self.assertEqual(system_cost(row, "selector_primary", params), 12 * 300)
        self.assertEqual(system_cost(row, "review_by_weaker", params, 4.0), 8 * 900 + 4 * 500)
        self.assertTrue({W, S} <= PARAMS_B.keys())

    def test_build_and_every_cost_card(self):
        db, art = make_world(self.tmp)
        out = self.tmp / "cost"
        self.run_script(ROOT / "scripts/build_cost_rows.py", "--db", db, "--artifacts", art, "--out-dir", out)
        report = json.loads((out / "reports/cost_pairs.json").read_text())
        self.assertEqual(len(report["pairs"]), 2)
        math = next(p for p in report["pairs"] if p["domain"] == "math")
        self.assertEqual(math["n"], 120)
        self.assertEqual(math["mean_tokens"]["review_by_s"], [300, 50])
        cost = math["cost_per_question"]["main"]
        self.assertAlmostEqual(cost["stronger_alone"], PARAMS_B[S] * 300)
        self.assertAlmostEqual(cost["review_by_stronger"], PARAMS_B[W] * 300 + PARAMS_B[S] * 350)
        self.assertEqual(math["jev"]["calls"], 20)
        self.assertIn("- 无", (out / "reports/summary.md").read_text())
        cards = [p for p in sorted((ROOT / "claims").glob("H*.yaml"))
                 if yaml.safe_load(p.read_text(encoding="utf-8")).get("analysis") == "cost"]
        self.assertGreaterEqual(len(cards), 2)
        for path in cards:
            card = cost_card(path, self.tmp)
            self.run_script(ROOT / "scripts/evaluate_cost_claims.py", "--cost", out, "--claim", card,
                            "--out-dir", self.tmp / "claims")
            cid = yaml.safe_load(card.read_text(encoding="utf-8"))["id"]
            result = json.loads((self.tmp / f"claims/{cid}_result.json").read_text())
            for comp in result["comparisons"]:
                self.assertEqual(sum(comp["counts"].values()), 2)
                self.assertNotIn("p_two_sided", comp["bootstrap"]["overall"])
            self.assertIn(cid, (self.tmp / f"claims/{cid}_summary.md").read_text())
        h10 = json.loads((self.tmp / "claims/H10_result.json").read_text())
        oracle = next(c for c in h10["comparisons"] if c["a"] == "oracle")
        self.assertGreaterEqual(oracle["bootstrap"]["overall"]["mean_diff_pp"], 0)
        self.assertAlmostEqual(oracle["cost_ratio"]["main"]["overall"], (PARAMS_B[W] + PARAMS_B[S]) / PARAMS_B[S])

    def test_missing_tokens_drop_the_pair_and_say_why(self):
        db, art = make_world(self.tmp, drop_usage=True)
        out = self.tmp / "cost"
        self.run_script(ROOT / "scripts/build_cost_rows.py", "--db", db, "--artifacts", art, "--out-dir", out)
        report = json.loads((out / "reports/cost_pairs.json").read_text())
        self.assertEqual([p["domain"] for p in report["pairs"]], ["math"])
        self.assertIn("code Qwen3-4B / Qwen3-8B: token counts missing", (out / "reports/summary.md").read_text())
        run = subprocess.run([sys.executable, str(ROOT / "scripts/build_cost_rows.py"), "--db", str(db),
                              "--artifacts", str(art), "--out-dir", str(out)], capture_output=True, text=True)
        self.assertNotEqual(run.returncode, 0)  # never overwrites


if __name__ == "__main__":
    unittest.main()
