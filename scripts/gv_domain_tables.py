#!/usr/bin/env python3
"""Archive-only GV tables. Run on the server; never copy problems or manifests locally.

Only grades and resolved configs are read. G is the actual reviewed source, V the
same reviewer's independent solve. Pair labels in YAML are NOT directional roles.
Development/test stay separate; pooled counts are descriptive, matched directions only.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import math
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "math": dict(zip(
        [f"c{i}" for i in range(200, 210)],
        ["c062", "c092", "c057", "c063", "c096", "c093", "c097", "c068", "c056", "c098"])),
    "knowledge": {"c070": "c081", "c071": "c082", "c074": "c120", "c075": "c121",
                  "c076": "c122", "c077": "c123", "c078": "c089", "c079": "c124", "c080": "c125"},
}
EVENTS = {
    "011": "V 本来会做，审阅后纠正 G", "010": "V 本来会做，审阅后却错了",
    "001": "两者本来都错，审阅后答对", "000": "两者本来都错，审阅后仍错",
    "111": "两者本来都对，审阅后仍对", "110": "两者本来都对，审阅后却错了",
    "100": "只有 G 本来答对，审阅后改错", "101": "只有 G 本来答对，审阅后保住正确",
}


def paired(a, b):
    n = len(a)
    win, loss = sum(x > y for x, y in zip(a, b)), sum(x < y for x, y in zip(a, b))
    delta = (win - loss) / n
    se = math.sqrt(max(win + loss - (win - loss) ** 2 / n, 0)) / n
    return {"win": win, "loss": loss, "rows": win-loss, "pp": 100*delta,
            "ci95_pp": [100*(delta-1.96*se), 100*(delta+1.96*se)]}


def measure(g, v, gv, stronger):
    if not (len(g) == len(v) == len(gv)) or not g:
        raise ValueError("empty or unequal input lengths")
    if any(x not in (0, 1) for xs in (g, v, gv) for x in xs):
        raise ValueError("non-binary outcome")
    cells = {k: 0 for k in EVENTS}
    for a, b, c in zip(g, v, gv):
        cells[f"{a}{b}{c}"] += 1
    base = g if stronger == "G" else v
    mx = g if sum(g) > sum(v) else v
    result = {"n": len(g), "G": sum(g), "V": sum(v), "GV": sum(gv), "events": cells,
              "baseline2_role": stronger, "baseline2": sum(base), "hindsight_max": sum(mx),
              "GV_minus_G": paired(gv, g), "GV_minus_baseline2": paired(gv, base),
              "GV_minus_hindsight_max": sum(gv)-sum(mx),
              "independent_oracle": sum(a or b for a,b in zip(g,v)),
              "exposed_oracle": sum(a or b for a,b in zip(g,gv))}
    assert result["GV_minus_G"]["rows"] == cells["011"]+cells["001"]-cells["110"]-cells["100"]
    assert sum(gv)-sum(v) == cells["101"]+cells["001"]-cells["010"]-cells["110"]
    return result


class Archives:
    def __init__(self, root):
        self.root, self.sources = root, {}

    def find(self, cid):
        hits = [p for p in self.root.glob(f"*__{cid}__*") if (p/"checksums.sha256").exists()
                and (p/"condition.resolved.yaml").exists()]
        if len(hits) > 1:
            raise ValueError(f"ambiguous archives for {cid}: {hits}")
        return hits[0] if hits else None

    def checked(self, run, rel):
        path = run / rel
        expected = {}
        for line in (run/"checksums.sha256").read_text().splitlines():
            h, name = line.split(maxsplit=1)
            expected[name.lstrip("*").removeprefix("./")] = h
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected.get(rel) != actual:
            raise ValueError(f"checksum mismatch or absent: {path}")
        self.sources[str(path)] = actual
        return path

    def config(self, run):
        return yaml.safe_load(self.checked(run, "condition.resolved.yaml").read_text())

    def grades(self, run, outcome="u0"):
        out = {}
        for line in self.checked(run, "records/grades.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r["outcome"] != outcome:
                continue
            pid = str(r["problem_id"])
            if pid in out or r["u"] not in (0, 1):
                raise ValueError(f"duplicate/null/nonbinary grade in {run}")
            out[pid] = r["u"]
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts", type=Path, default=ROOT/"artifacts")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    out = args.out_dir/"reports"
    out.mkdir(parents=True, exist_ok=False)
    ar = Archives(args.artifacts)
    results, headrooms, missing, pooled = [], [], [], []
    for domain, mapping in MODELS.items():
        solos, by_model = {}, {}
        for dc, tc in mapping.items():
            dr, tr = ar.find(dc), ar.find(tc)
            if not dr or not tr:
                missing.append(f"{domain} solo {dc}/{tc}")
                continue
            config = ar.config(dr)
            name = config["model"]["model_id"]
            solos[dc] = {"name": name, "runs": {"dev": dr, "test": tr},
                         "grades": {"dev": ar.grades(dr), "test": ar.grades(tr)}}
            by_model[name] = dc
        for a,b in itertools.combinations(solos, 2):
            a,b = sorted((a,b), key=lambda k: (sum(solos[k]["grades"]["dev"].values()), k))
            for split in ("dev", "test"):
                ga,gb = (solos[k]["grades"][split] for k in (a,b))
                if set(ga) != set(gb):
                    raise ValueError("solo membership mismatch")
                oracle = sum(ga[p] or gb[p] for p in ga)
                headrooms.append(dict(domain=domain, split=split, g=solos[a]["name"], v=solos[b]["name"],
                                      n=len(ga), G=sum(ga.values()), V=sum(gb.values()), oracle=oracle,
                                      headroom_vs_baseline2=oracle-sum(gb.values()),
                                      headroom_vs_hindsight=oracle-max(sum(ga.values()),sum(gb.values()))))
        pairfile = "mathfull_testpool_dev1000.yaml" if domain == "math" else "mmlupro_dev1000.yaml"
        pairs = yaml.safe_load((ROOT/"pairs"/pairfile).read_text())["pairs"]
        if domain == "knowledge":
            # New 72B pairs are registered in conditions and the study plan, not the older pair YAML.
            pairs += [{"id": f"extra-{g}", "g": {"condition": g}, "v": {"condition": "c080"},
                       "gv": [{"condition": str(c), "reviewer": side} for c,side in dirs]}
                      for g,dirs in [("c076", [("c270","v"),("c271","g")]),
                                     ("c074", [("c272","v"),("c273","g")])]]
        for pair in pairs:
            a,b = pair["g"]["condition"], pair["v"]["condition"]
            if a not in solos or b not in solos:
                continue
            stronger_c = max((a,b), key=lambda k: (sum(solos[k]["grades"]["dev"].values()), k))
            for idx, direction in enumerate(pair["gv"]):
                dc = direction["condition"]
                if domain == "math":
                    tc = pair["test_gv"][idx]
                    if tc == "c244": tc = "c258"
                else:
                    tc = ({"c072":"c083", "c073":"c084", "c085":"c136", "c086":"c137",
                           "c087":"c176", "c088":"c177"}.get(dc)
                          or f"c{int(dc[1:])+ (12 if 126 <= int(dc[1:]) <= 135 else 8 if 170 <= int(dc[1:]) <= 175 else 4)}")
                arrays = {}
                for split,cid in (("dev",dc),("test",tc)):
                    rr = ar.find(cid)
                    if rr is None:
                        missing.append(f"{domain} {pair['id']} {split} {cid}")
                        continue
                    conf = ar.config(rr)
                    gr = args.artifacts / Path(conf["generator_source"]["artifact_path"]).name
                    reviewer = by_model[conf["model"]["model_id"]]
                    generator = a if reviewer == b else b
                    # Match original whole-pool G/V realizations for c042/c043, not reruns c062/c092.
                    vr = ar.find("c041" if cid == "c042" else "c040") if cid in ("c042","c043") else solos[reviewer]["runs"][split]
                    if ar.config(gr)["model"]["model_id"] != solos[generator]["name"]:
                        raise ValueError(f"{cid}: wrong reviewed source {gr.name}; expected {solos[generator]['name']}; reviewer {conf['model']['model_id']}")
                    ids = set(solos[generator]["grades"][split])
                    yg,yv,yy = ar.grades(gr),ar.grades(vr),ar.grades(rr,"u_gv")
                    if not all(ids <= set(x) for x in (yg,yv,yy)):
                        raise ValueError(f"incomplete direction {cid}")
                    ids = sorted(ids)
                    vectors = [[d[p] for p in ids] for d in (yg,yv,yy)]
                    role = "G" if stronger_c == generator else "V"
                    row = dict(domain=domain, pair=pair["id"], split=split, condition=cid,
                               generator=solos[generator]["name"], reviewer=solos[reviewer]["name"],
                               runs=[gr.name,vr.name,rr.name], **measure(*vectors,role))
                    results.append(row)
                    arrays[split] = (ids,vectors,row)
                if len(arrays) == 2:
                    di,dv,dr = arrays["dev"]; ti,tv,tr = arrays["test"]
                    if set(di) & set(ti): raise ValueError("dev/test overlap")
                    pooled.append(dict(domain=domain,pair=pair["id"],split="pooled-descriptive",
                                       condition=f"{dc}/{tc}",generator=dr["generator"],reviewer=dr["reviewer"],
                                       **measure(*[x+y for x,y in zip(dv,tv)],dr["baseline2_role"])))
    report = {"directions": results, "pooled": pooled, "headrooms": headrooms, "missing": missing,
              "source_sha256": ar.sources, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/"tables.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    lines = ["# 数学与知识域 GV 总表", "", "G = 实际被审者，V = 审阅者独立作答。差值以题数表示。B2 为开发集选定的较强单模型；max 为本表样本上的事后较强单模型。",
             "所有表为归档分析；pooled 仅合并两侧均齐的方向，是描述性汇总。不同方向重复使用题目，跨方向事件数不是独立题数。不对加总事件做独立样本检验。",
             "c042/c043 使用原始 c040/c041 的同题结果，与两两上限表 c062/c092 的另次生成不同。", ""]
    for group in ("dev","test","pooled-descriptive"):
        rs = [r for r in results+pooled if r["split"] == group]
        lines += [f"## {group}", "", "|域|条件|G → V|N|G|V|GV|GV−G|GV−B2|GV−max|独立并集|审阅后并集|",
                  "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in rs:
            lines.append(f"|{r['domain']}|{r['condition']}|{r['generator'].split('/')[-1]} → {r['reviewer'].split('/')[-1]}|{r['n']}|{r['G']}|{r['V']}|{r['GV']}|{r['GV_minus_G']['rows']:+d}|{r['GV_minus_baseline2']['rows']:+d}|{r['GV_minus_hindsight_max']:+d}|{r['independent_oracle']}|{r['exposed_oracle']}|")
        lines += ["", "|域|方向数|GV>G|GV>B2|GV>事后max|", "|---|---:|---:|---:|---:|"]
        for domain in MODELS:
            sub = [r for r in rs if r["domain"] == domain]
            lines.append(f"|{domain}|{len(sub)}|{sum(r['GV_minus_G']['rows']>0 for r in sub)}|{sum(r['GV_minus_baseline2']['rows']>0 for r in sub)}|{sum(r['GV_minus_hindsight_max']>0 for r in sub)}|")
        lines += ["", "|域|条件|"+"|".join(EVENTS)+"|", "|---|---|"+"---:|"*8]
        for r in rs:
            lines.append(f"|{r['domain']}|{r['condition']}|"+"|".join(str(r['events'][k]) for k in EVENTS)+"|")
        lines.append("")
    lines += ["## 八种事件", "", "|G,V,GV|称呼|", "|---|---|"]
    lines += [f"|{k}|{v}|" for k,v in EVENTS.items()]
    lines += ["", "011 可作为能力迁移的操作性计数，不能单独证明因果机制；010 不能仅凭对错确认跟随 G。101 不一定是主动采纳，也可能由解析失败回退产生。", "",
              "## 两两上限（测试集）", "", "|域|开发集较弱 / 较强|N|G|V=B2|并集|上限−B2|上限−事后max|", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in headrooms:
        if r["split"] == "test":
            lines.append(f"|{r['domain']}|{r['g'].split('/')[-1]} / {r['v'].split('/')[-1]}|{r['n']}|{r['G']}|{r['V']}|{r['oracle']}|{r['headroom_vs_baseline2']}|{r['headroom_vs_hindsight']}|")
    lines += ["", "## 未完成方向", ""]+[f"- {m}" for m in missing]
    (out/"summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"directions":len(results),"pooled":len(pooled),"headroom_rows":len(headrooms),"missing":missing}))


if __name__ == "__main__":
    main()
