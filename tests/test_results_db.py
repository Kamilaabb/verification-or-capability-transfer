"""End-to-end check of build_results_db.py and evaluate_claim.py on fake archives."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_results_db", ROOT / "scripts/build_results_db.py")
db = importlib.util.module_from_spec(spec)
spec.loader.exec_module(db)

LABELS = {
    "manifests": [
        {"dataset": "ds-math", "manifest": "dev", "domain": "math", "split": "dev"},
        {"dataset": "ds-math", "manifest": "test", "domain": "math", "split": "test"},
        {"dataset": "ds-math", "manifest": "pool", "domain": "math", "split": "test",
         "solo_table": False, "restrict_to": "test"},
        {"dataset": "ds-code", "manifest": "dev", "domain": "code", "split": "dev"},
    ],
    "exclude": ["c099"],
    "prefer": [],
    "stronger_from": {"math": "dev", "code": "dev"},
    "domains": {
        "math": {"tiers": {"strong": ["Big"], "weak": ["Small", "Mid"]},
                 "design_pairs": [{"models": ["Big", "Small"], "cell": "strong-weak", "headroom": "high"}]},
        "code": {"tiers": {}, "design_pairs": [{"models": ["Big", "Small"], "cell": "x", "headroom": "low"}]},
    },
}


class FakeArchives:
    def __init__(self, root: Path):
        self.root, self.n = root, 0

    def make(self, cid, dataset, manifest, model, grades, outcome="u0", source=None):
        self.n += 1
        run = self.root / f"2026T{self.n:04d}__{cid}__x__r"
        (run / "records").mkdir(parents=True)
        cfg = {"condition_id": cid, "model": {"model_id": f"org/{model}"},
               "dataset": {"path": f"datasets/{dataset}/dataset.yaml", "manifest": manifest}}
        if source is not None:
            cfg["generator_source"] = {"artifact_path": f"/server/artifacts/{source.name}"}
        (run / "condition.resolved.yaml").write_text(yaml.safe_dump(cfg))
        (run / "records/grades.jsonl").write_text(
            "".join(json.dumps({"problem_id": p, "outcome": outcome, "u": u}) + "\n" for p, u in grades.items()))
        sums = [f"{hashlib.sha256((run / rel).read_bytes()).hexdigest()}  {rel}"
                for rel in ("condition.resolved.yaml", "records/grades.jsonl")]
        (run / "checksums.sha256").write_text("\n".join(sums) + "\n")
        return run


def vec(ids, p, rng):
    return {i: int(rng.random() < p) for i in ids}


class ResultsDatabase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        rng = random.Random(0)
        fa = FakeArchives(self.tmp / "artifacts")
        dev, test = [f"d{i}" for i in range(200)], [f"t{i}" for i in range(300)]
        pool = test + [f"x{i}" for i in range(50)]
        big_dev = fa.make("c001", "ds-math", "dev", "Big", vec(dev, .8, rng))
        small_dev = fa.make("c002", "ds-math", "dev", "Small", vec(dev, .4, rng))
        fa.make("c003", "ds-math", "test", "Big", vec(test, .8, rng))
        fa.make("c004", "ds-math", "test", "Small", vec(test, .4, rng))
        big_pool = fa.make("c005", "ds-math", "pool", "Big", vec(pool, .8, rng))
        small_pool = fa.make("c006", "ds-math", "pool", "Small", vec(pool, .4, rng))
        # Stronger side generating does better on every split.
        fa.make("c010", "ds-math", "dev", "Small", vec(dev, .8, rng), "u_gv", source=big_dev)
        fa.make("c011", "ds-math", "dev", "Big", vec(dev, .6, rng), "u_gv", source=small_dev)
        fa.make("c012", "ds-math", "pool", "Small", vec(pool, .8, rng), "u_gv", source=big_pool)
        fa.make("c013", "ds-math", "pool", "Big", vec(pool, .6, rng), "u_gv", source=small_pool)
        fa.make("c099", "ds-math", "dev", "Big", vec(dev, .0, rng), "u_gv", source=small_dev)
        cb = fa.make("c020", "ds-code", "dev", "Big", vec(dev, .7, rng))
        cs = fa.make("c021", "ds-code", "dev", "Small", vec(dev, .3, rng))
        fa.make("c022", "ds-code", "dev", "Small", vec(dev, .7, rng), "u_gv", source=cb)
        fa.make("c023", "ds-code", "dev", "Big", vec(dev, .5, rng), "u_gv", source=cs)
        fa.make("c030", "ds-other", "dev", "Big", vec(dev, .5, rng))
        self.artifacts = self.tmp / "artifacts"

    def test_build_and_evaluate(self):
        result = db.build(self.artifacts, LABELS)
        self.assertEqual(result["ambiguous"], [])
        self.assertEqual(result["missing_design"], [])
        math = result["pairs"]["math"][0]
        self.assertEqual(math["stronger"], "Big")
        self.assertTrue(math["in_design_sample"])
        self.assertEqual(math["directions"]["G=Big"], {"dev": "c010", "test": "c012"})
        test_rows = [r for r in result["dir_rows"] if r["condition"] == "c012"]
        self.assertEqual(len(test_rows), 300)  # restricted to the test split, pool-only rows dropped
        self.assertTrue(all(r["v_condition"] == "c006" for r in test_rows))  # V from the same manifest
        self.assertTrue(any("c099" in x for x in result["ignored"]))
        self.assertTrue(any("ds-other / dev" in x and "1 个归档" in x for x in result["ignored"]))

        out = self.tmp / "db_out"
        db.write(result, out)
        self.assertTrue((out / "reports/summary.md").read_text().startswith("# 结果库清单"))
        card = yaml.safe_load((ROOT / "claims/H01_stronger_generates.yaml").read_text())
        card["splits"] = {"math": ["dev", "test"], "code": ["dev"]}
        card["bootstrap"] = {"reps": 200, "seed": 1}
        card_path = self.tmp / "card.yaml"
        card_path.write_text(yaml.safe_dump(card, allow_unicode=True))
        subprocess.run([sys.executable, str(ROOT / "scripts/evaluate_claim.py"), "--db", str(out),
                        "--claim", str(card_path), "--out-dir", str(self.tmp / "claim_out")], check=True,
                       capture_output=True)
        report = json.loads((self.tmp / "claim_out/reports/result.json").read_text())
        primary = report["analyses"]["主分析：设计配对"]["report"]
        self.assertEqual(primary["sign_test"]["units"], 2)
        sizes = {u["id"]: u["n"] for u in primary["units"]}
        self.assertEqual(sizes, {"math:Big>Small": 500, "code:Big>Small": 200})
        self.assertGreater(primary["bootstrap"]["overall"]["mean_diff_pp"], 0)
        summary = (self.tmp / "claim_out/reports/summary.md").read_text()
        self.assertIn("写卡片时已看过的域", summary)

    def test_unregistered_manifest_of_a_labelled_dataset_is_flagged(self):
        fa = FakeArchives(self.artifacts)
        fa.n = 300
        fa.make("c060", "ds-code", "development", "Big", {"d0": 1})
        result = db.build(self.artifacts, LABELS)
        self.assertTrue(any("ds-code / development" in x and "清单名未登记" in x for x in result["ignored"]))
        known = {**LABELS, "out_of_scope": [{"dataset": "ds-code", "manifest": "development", "reason": "pilot"}]}
        result = db.build(self.artifacts, known)
        self.assertTrue(any("ds-code / development" in x and "已知不纳入：pilot" in x for x in result["ignored"]))

    def test_pairs_with_different_question_sets_keep_their_own_questions(self):
        mod_spec = importlib.util.spec_from_file_location("evaluate_claim", ROOT / "scripts/evaluate_claim.py")
        ev = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(ev)
        rows = []
        for gen, rev, ids in (("A", "B", range(10)), ("B", "A", range(10)), ("C", "D", range(9)), ("D", "C", range(9))):
            rows += [{"domain": "m", "split": "dev", "generator": gen, "reviewer": rev, "problem_id": str(i),
                      "gv": int(gen in "AC")} for i in ids]
        pairs = {"m": [{"models": ["A", "B"], "stronger": "A", "in_design_sample": True},
                       {"models": ["C", "D"], "stronger": "C", "in_design_sample": True}]}
        units, notes = ev.stronger_generates_minus_stronger_reviews(rows, pairs, {"m": ["dev"]}, True)
        self.assertEqual([u.n for u in units], [10, 9])
        self.assertTrue(any("A>B 用 10 道" in n for n in notes))

    def test_every_claim_card_runs(self):
        result = db.build(self.artifacts, LABELS)
        out = self.tmp / "db_all"
        db.write(result, out)
        for path in sorted((ROOT / "claims").glob("H*.yaml")):
            card = yaml.safe_load(path.read_text())
            card["splits"] = {"math": ["dev", "test"], "code": ["dev"]}
            card["bootstrap"] = {"reps": 50, "seed": 1}
            card_path = self.tmp / path.name
            card_path.write_text(yaml.safe_dump(card, allow_unicode=True))
            target = self.tmp / f"out_{card['id']}"
            if card.get("analysis") == "cost":
                continue  # needs token counts; run end to end in tests/test_cost_claims.py
            if card.get("analysis") == "outcome_levels":
                # H09 reads only the conditions card's per-direction report (H08 sorts before H09).
                h08 = next(self.tmp.glob("out_H08/reports/result.json"))
                run = subprocess.run([sys.executable, str(ROOT / "scripts/explain_outcome_levels.py"), "--h08", str(h08),
                                      "--card", str(card_path), "--out-dir", str(target)], capture_output=True, text=True)
                self.assertEqual(run.returncode, 0, run.stderr)
                self.assertIn(card["id"], (target / "H09_summary.md").read_text())
                continue
            script = "explain_conditions.py" if card.get("analysis") == "conditions" else "evaluate_claim.py"
            subprocess.run([sys.executable, str(ROOT / "scripts" / script), "--db", str(out),
                            "--claim", str(card_path), "--out-dir", str(target)], check=True, capture_output=True)
            self.assertIn(card["id"], (target / "reports/summary.md").read_text())

    def test_direction_estimands_count_the_right_questions(self):
        spec_ = importlib.util.spec_from_file_location("evaluate_claim", ROOT / "scripts/evaluate_claim.py")
        ev = importlib.util.module_from_spec(spec_)
        spec_.loader.exec_module(ev)
        # (g, v, gv) per question for one direction: S (stronger) reviews W.
        triples = [(0, 0, 1), (1, 1, 0), (0, 1, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1), (0, 0, 0), (0, 1, 0)]
        rows = [{"domain": "m", "split": "dev", "generator": "W", "reviewer": "S", "problem_id": str(i),
                 "g": g, "v": v, "gv": gv} for i, (g, v, gv) in enumerate(triples)]
        pairs = {"m": [{"models": ["S", "W"], "stronger": "S", "in_design_sample": True}]}
        sp = {"m": ["dev"]}
        def one(name):
            units, _ = ev.ESTIMANDS[name](rows, pairs, sp, True)
            self.assertEqual(len(units), 1)
            return units[0]
        u = one("gv_minus_g")                     # GV 4/8 vs G 3/8
        self.assertAlmostEqual(u.diff_pp, 12.5)
        u = one("gv_minus_stronger_solo")         # GV 4/8 vs V (stronger) 5/8
        self.assertAlmostEqual(u.diff_pp, -12.5)
        u = one("agreement_gain_minus_loss")      # (0,0,1) once vs (1,1,0) once
        self.assertAlmostEqual(u.diff_pp, 0.0)
        self.assertEqual((sum(u.a), sum(u.b)), (1, 1))
        u = one("follows_generator_on_disagreement")  # disagreement rows: 3 follow G, 1 follows V
        self.assertEqual(u.n, 4)
        self.assertEqual((sum(u.a), sum(u.b)), (3, 1))
        u = one("exactly_one_right_before_minus_after")  # before 4, after (g != gv) 3
        self.assertEqual((sum(u.a), sum(u.b)), (4, 3))
        u = one("oracle_minus_stronger_solo")     # oracle 6/8 vs stronger 5/8
        self.assertEqual((sum(u.a), sum(u.b)), (6, 5))

    def test_conditions_measure_each_direction(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import explain_conditions as ec
        triples = [(0, 0, 1), (1, 1, 0), (0, 1, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1), (0, 0, 0), (0, 1, 0)]
        rows = [{"domain": "m", "split": "dev", "generator": "W", "reviewer": "S", "problem_id": str(i),
                 "g": g, "v": v, "gv": gv} for i, (g, v, gv) in enumerate(triples)]
        pairs = {"m": [{"models": ["S", "W"], "stronger": "S", "in_design_sample": True,
                        "tiers": ["strong", "weak"], "cell": None, "headroom": None}]}
        directions, pair_items, _ = ec.measure(rows, pairs, {"m": ["dev"]}, True)
        rec = directions[0]["item"].record()
        # S alone 5/8, W alone 3/8, GV 4/8; W right & S wrong once; exactly one right 4 times.
        self.assertEqual((rec["dG"], rec["dC"], rec["gap"], rec["head"], rec["dis"]), (12.5, -12.5, 25.0, 12.5, 50.0))
        self.assertAlmostEqual(rec["head"], (rec["dis"] - rec["gap"]) / 2)
        self.assertEqual(directions[0]["kind"], "stronger_reviews")
        self.assertEqual(directions[0]["category"], "white")
        self.assertEqual(directions[0]["cell"], "strong-weak")
        self.assertEqual(pair_items[0]["item"].value("gap"), 25.0)

    def test_conditions_recover_a_planted_slope(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import explain_conditions as ec
        rng = random.Random(5)
        rows, pairs = [], {}
        # Each pair: S right with prob ps, W with prob pw; review keeps G's answer except it copies the
        # stronger reviewer's on a fixed share of questions, so Δ_C falls as the gap grows.
        for domain in ("m", "k"):
            pairs[domain] = []
            for j, (ps, pw) in enumerate([(.9, .3), (.8, .5), (.7, .6), (.85, .4), (.75, .7), (.95, .2)]):
                s_name, w_name = f"S{domain}{j}", f"W{domain}{j}"
                pairs[domain].append({"models": [s_name, w_name], "stronger": s_name, "in_design_sample": True,
                                      "tiers": ["strong", "weak"], "cell": None, "headroom": None})
                for i in range(400):
                    s, w = int(rng.random() < ps), int(rng.random() < pw)
                    gv_sw = s if rng.random() < .5 else w      # S reviews W
                    gv_ws = s                                   # W reviews S: keeps S
                    rows.append({"domain": domain, "split": "dev", "generator": w_name, "reviewer": s_name,
                                 "problem_id": str(i), "g": w, "v": s, "gv": gv_sw})
                    rows.append({"domain": domain, "split": "dev", "generator": s_name, "reviewer": w_name,
                                 "problem_id": str(i), "g": s, "v": w, "gv": gv_ws})
        card = {"splits": {"m": ["dev"], "k": ["dev"]}, "bootstrap": {"reps": 100, "seed": 1}}
        res = ec.analyse(rows, pairs, card, True)
        gap_slope = res["fits"]["stronger_reviews"]["statistics"]["合计|dC|gap|gap"]
        self.assertAlmostEqual(gap_slope["point"], -5.0, delta=1.0)   # Δ_C ≈ −gap/2 → −5 per 10 pp
        self.assertLess(gap_slope["ci95"][1], 0)
        self.assertLess(res["correlation"]["statistics"]["合计"]["point"], 0)
        text = ec.render({"id": "H08", "statement_zh": "x"}, {"主分析": res})
        self.assertIn("能力差距和上限空间的关系", text)

    def test_bad_grade_file_is_reported_not_fatal(self):
        fa = FakeArchives(self.artifacts)
        fa.n = 200
        bad = fa.make("c050", "ds-code", "dev", "Odd", {"d0": 1})
        lines = (bad / "records/grades.jsonl").read_text()
        (bad / "records/grades.jsonl").write_text(lines + lines)  # duplicate row
        sums = [f"{hashlib.sha256((bad / rel).read_bytes()).hexdigest()}  {rel}"
                for rel in ("condition.resolved.yaml", "records/grades.jsonl")]
        (bad / "checksums.sha256").write_text("\n".join(sums) + "\n")
        result = db.build(self.artifacts, LABELS)
        self.assertTrue(any("__c050__" in x and "1 duplicate rows" in x for x in result["problems"]))
        self.assertEqual(result["missing_design"], [])

    def test_duplicate_directions_are_ambiguous_unless_preferred(self):
        fa = FakeArchives(self.artifacts)
        fa.n = 100
        src = next(self.artifacts.glob("*__c001__*"))
        fa.make("c040", "ds-math", "dev", "Small", {"d0": 1}, "u_gv", source=src)
        result = db.build(self.artifacts, LABELS)
        self.assertTrue(any("c040" in x and "c010" in x for x in result["ambiguous"]))
        result = db.build(self.artifacts, {**LABELS, "prefer": ["c010"]})
        self.assertFalse(any("c040" in x for x in result["ambiguous"]))


if __name__ == "__main__":
    unittest.main()
