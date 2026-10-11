"""RQ2: what review does on each kind of question, derived from the H02-H08 claim reports.

For every review direction the questions split by whether the reviewed model (G) and the
reviewer (V) are right when answering alone. The claim reports give enough counts to
recover how often the reviewed answer (GV) ends right in each cell:

  both right     -> broken     = H04 b_correct
  both wrong     -> fixed      = H04 a_correct
  only G right   -> kept right = r_a
  only V right   -> adopted    = r_b

with n_a + n_b = H06 discordant, n_a - n_b = acc(G) - acc(V), H05 a_correct = r_a + (n_b - r_b)
(GV agrees with G), and acc(GV) = (both right - broken) + fixed + r_a + r_b.
Needs no data beyond the committed claim JSONs.
"""
import json
import random
import sys
from pathlib import Path

CLAIMS = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/analysis/claims/20261004")
OUT = CLAIMS / "rq2_cells.json"
MAIN = "主分析：设计配对"


def units(h):
    d = json.loads((CLAIMS / f"{h}_result.json").read_text())
    return {u["id"]: u for u in d["analyses"][MAIN]["report"]["units"]}


h02, h03, h04, h05, h06 = (units(h) for h in ("H02", "H03", "H04", "H05", "H06"))
h08 = json.loads((CLAIMS / "H08_result.json").read_text())["analyses"][MAIN]["directions"]

rows = []
for x in h08:
    i, n = x["id"], x["n"]
    gv, g = h02[i]["a_correct"], h02[i]["b_correct"]
    if x["kind"] == "stronger_reviews":
        v = h03[i]["b_correct"]
    else:
        v = g - round(x["gap"] * n / 100)
    dis = h06[i]["a_correct"]
    n_a, n_b = (dis + (g - v)) // 2, (dis - (g - v)) // 2
    assert n_a + n_b == dis and n_a - n_b == g - v, i
    fixed, broken = h04[i]["a_correct"], h04[i]["b_correct"]
    both_right = g - n_a
    follow_g = h05[i]["a_correct"]
    s = gv - (both_right - broken) - fixed
    dd = follow_g - n_b
    assert (s + dd) % 2 == 0, i
    r_a, r_b = (s + dd) // 2, (s - dd) // 2
    assert 0 <= r_a <= n_a and 0 <= r_b <= n_b, i
    rows.append(dict(
        id=i, domain=x["domain"], kind=x["kind"], gap=x["gap"], category=x["category"], worse=x["worse"], n=n,
        only_g_right=n_a, only_v_right=n_b, kept=r_a, adopted=r_b, fixed=fixed, broken=broken,
        keep_rate=100 * r_a / n_a if n_a else None, adopt_rate=100 * r_b / n_b if n_b else None,
        follow_g=100 * follow_g / dis,
        # contributions to acc(GV) - acc(G) and acc(GV) - acc(V), in points over all questions
        gain_from_v=100 * r_b / n, loss_from_g=100 * (n_a - r_a) / n, net_both=100 * (fixed - broken) / n,
        dC=x["dC"], dG=x["dG"], head=x["head"],
        # best of two answers per question, relative to the stronger model alone: G and V answering alone,
        # and G alone with the reviewed answer GV (both wrong after review = still-both-wrong + adopted-not)
        oracle_before=100 * (n - (n - both_right - dis) - max(g, v)) / n,
        oracle_after=100 * (n - (n - both_right - dis - fixed) - (n_b - r_b) - max(g, v)) / n,
        # share of questions where exactly one model is right: alone (G vs V) and after review (GV vs V alone)
        dis_before=100 * dis / n, dis_after=100 * h06[i]["b_correct"] / n,
    ))


def eq_mean(sub, key):
    doms = sorted({r["domain"] for r in sub})
    per = [sum(r[key] for r in sub if r["domain"] == d) / sum(1 for r in sub if r["domain"] == d) for d in doms]
    return sum(per) / len(per)


def boot(sub, key, reps=2000, seed=20261003):
    rnd = random.Random(seed)
    by = {}
    for r in sub:
        by.setdefault(r["domain"], []).append(r)
    stats = []
    for _ in range(reps):
        samp = [rnd.choice(v) for v in by.values() for _ in v]
        stats.append(eq_mean(samp, key))
    stats.sort()
    return [stats[int(0.025 * reps)], stats[int(0.975 * reps) - 1]]


summary = {}
for kind in ("stronger_reviews", "weaker_reviews"):
    sub = [r for r in rows if r["kind"] == kind]
    summary[kind] = {"directions": len(sub)}
    for key in ("follow_g", "keep_rate", "adopt_rate", "gain_from_v", "loss_from_g", "net_both",
                "oracle_before", "oracle_after", "dC"):
        summary[kind][key] = {"mean": eq_mean(sub, key), "ci95": boot(sub, key)}
    summary[kind]["per_domain_adopt"] = {d: eq_mean([r for r in sub if r["domain"] == d], "adopt_rate")
                                         for d in sorted({r["domain"] for r in sub})}
    summary[kind]["per_domain_keep"] = {d: eq_mean([r for r in sub if r["domain"] == d], "keep_rate")
                                        for d in sorted({r["domain"] for r in sub})}

OUT.write_text(json.dumps({"source": str(CLAIMS), "summary": summary, "directions": rows},
                          ensure_ascii=False, indent=1))
for k, v in summary.items():
    print(k, v["directions"])
    for key in ("follow_g", "keep_rate", "adopt_rate", "gain_from_v", "loss_from_g", "net_both",
                "oracle_before", "oracle_after", "dC"):
        print(f"  {key:12s} {v[key]['mean']:6.1f} [{v[key]['ci95'][0]:.1f}, {v[key]['ci95'][1]:.1f}]")
    print("  adopt by domain", {d: round(a, 1) for d, a in v["per_domain_adopt"].items()})
    print("  keep by domain ", {d: round(a, 1) for d, a in v["per_domain_keep"].items()})
