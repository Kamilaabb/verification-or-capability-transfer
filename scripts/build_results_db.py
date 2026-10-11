#!/usr/bin/env python3
"""Build the per-question results database from validated archives. Run on the server only.

Archives are discovered from their own condition.resolved.yaml: dataset + manifest pick the domain
and split (claims/labels.yaml), a `generator_source` makes the archive a review direction, otherwise it
is a solo run. Every file read is checksum-verified through gv_domain_tables.Archives. Problems and
manifests are never read; only grades and resolved configs.

Writes, into a new --out-dir:
  db/solo.jsonl        domain, split, model, problem_id, correct, condition
  db/directions.jsonl  domain, split, generator, reviewer, problem_id, g, v, gv, condition
  db/pairs.json        per domain: models, stronger side, design cell, in_design_sample, directions
  db/systems.jsonl     selector_primary, selector_secondary, JEV; u0-joined decisions
  reports/summary.md   inventory, missing design directions, ambiguous and ignored archives
  reports/inventory.json  source hashes
The db/ files hold problem ids and stay on the server; reports/ is safe to commit.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("gv_domain_tables", ROOT / "scripts/gv_domain_tables.py")
_gv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gv)
Archives = _gv.Archives


def short(model_id: str) -> str:
    return model_id.rstrip("/").split("/")[-1]


def has_file(run: Path, rel: str) -> bool:
    names = {line.split(maxsplit=1)[1].lstrip("*").removeprefix("./")
             for line in (run / "checksums.sha256").read_text().splitlines() if line.strip()}
    return rel in names


def read_grades(ar: Archives, run: Path, outcome: str) -> tuple[dict, str | None]:
    """Grades for one outcome, or a description of why the archive cannot be used.

    A bad archive is reported in the summary and left out, never repaired or guessed around,
    and one bad archive does not stop the rest of the database from being built.
    """
    try:
        return ar.grades(run, outcome), None
    except ValueError:
        pass
    seen, dup, null, other = set(), 0, 0, 0
    for line in (run / "records/grades.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("outcome") != outcome:
            continue
        pid = str(r["problem_id"])
        dup += pid in seen
        seen.add(pid)
        null += r.get("u") is None
        other += r.get("u") not in (0, 1, None)
    return {}, (f"{run.name}: outcome {outcome} has {len(seen)} problems, {dup} duplicate rows, "
                f"{null} null grades, {other} non-0/1 grades; archive left out")


def discover(artifacts: Path, labels: dict, ar: Archives) -> tuple[list[dict], list[str]]:
    manifests = {(m["dataset"], m["manifest"]): m for m in labels["manifests"]}
    exclude = set(labels.get("exclude", []))
    found, ignored, unmatched = [], [], defaultdict(int)
    for run in sorted(p for p in artifacts.iterdir() if p.is_dir()):
        if not ((run / "checksums.sha256").exists() and (run / "condition.resolved.yaml").exists()):
            continue
        cfg = ar.config(run)
        cid = cfg.get("condition_id") or run.name.split("__")[1]
        dataset = cfg.get("dataset") or {}
        key = (Path(str(dataset.get("path", ""))).parent.name, dataset.get("manifest"))
        if cid in exclude:
            ignored.append(f"{run.name}: excluded condition {cid}")
            continue
        if key not in manifests:
            unmatched[key] += 1  # other studies are out of scope, but every skip is counted in the summary
            continue
        if not has_file(run, "records/grades.jsonl") or not (cfg.get("model") or {}).get("model_id"):
            ignored.append(f"{run.name}: no grades or no model")
            continue
        entry = manifests[key]
        item = {"run": run, "condition": cid, "dataset": key[0], "manifest": key[1],
                "domain": entry["domain"], "split": entry["split"], "entry": entry,
                "model": short(cfg["model"]["model_id"]), "kind": "solo", "g_archive": None, "generator": None}
        source = cfg.get("generator_source") or {}
        if source.get("artifact_path"):
            g_run = artifacts / Path(source["artifact_path"]).name
            if not (g_run / "checksums.sha256").exists():
                ignored.append(f"{run.name}: reviewed source archive missing ({g_run.name})")
                continue
            item.update(kind="direction", g_archive=g_run, generator=short(ar.config(g_run)["model"]["model_id"]))
        found.append(item)
    labelled = {d for d, _ in manifests}
    known = {(o["dataset"], o["manifest"]): o["reason"] for o in labels.get("out_of_scope", [])}
    for (dataset, manifest), n in sorted(unmatched.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
        if (dataset, manifest) in known:
            flag = f"（已知不纳入：{known[(dataset, manifest)]}）"
        else:
            flag = "【数据集已登记、清单名未登记，请检查 labels.yaml】" if dataset in labelled else ""
        ignored.append(f"未登记的 (数据集, 清单) {dataset or '无'} / {manifest or '无'}：{n} 个归档 {flag}".rstrip())
    return found, ignored


def choose(groups: dict, prefer: set) -> tuple[dict, list[str]]:
    chosen, ambiguous = {}, []
    for key, cands in groups.items():
        if len(cands) == 1:
            chosen[key] = cands[0]
            continue
        preferred = [c for c in cands if c["condition"] in prefer]
        if len(preferred) == 1:
            chosen[key] = preferred[0]
        else:
            ambiguous.append(f"{key}: {sorted(c['condition'] for c in cands)}")
    return chosen, ambiguous


def ingest_systems(ar: Archives, labels: dict, registry: dict) -> tuple[list, list]:
    """Join archived decisions to u0 by ID, rejecting any discrepancy before writing.

    The registry pins legacy files that have no checksum sidecar. Existing sidecars
    are checked as well. Only grades, configs, decisions and aggregate reports are read.
    """
    rows, summaries, seen = [], [], set()
    entries = {(m['dataset'], m['manifest']): m for m in labels['manifests']}
    for group in registry['groups']:
        root = Path(group['root'])
        hashes = group['file_sha256']

        def checked(path):
            rel = str(path.relative_to(root)).replace('\\', '/')
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if hashes.get(rel) != actual:
                raise ValueError(f'registered system checksum mismatch: {path}')
            # Retain the original checksum gate wherever one was written.
            for parent in (path.parent.parent, root):
                if (parent / 'checksums.sha256').exists():
                    ar.checked(parent, str(path.relative_to(parent)).replace('\\', '/'))
                    break
            ar.sources[str(path)] = actual
            return path

        reports = sorted(root.glob(group['report_glob']))
        if len(reports) != group['pairs']:
            raise ValueError(f"{root.name}: expected {group['pairs']} complete pair reports; found {len(reports)}")
        used_files = set()
        for report_path in reports:
            report = json.loads(checked(report_path).read_text(encoding='utf-8'))
            pair_run = report_path.parent.parent
            checked(pair_run / 'reports/summary.md')
            domain, n = group['domain'], group['n']
            split = 'test' if group['evaluation'] == 'test' else 'dev'
            if group['evaluation'] not in ('test', 'cv') or report['domain'] != domain or report['n'] != n:
                raise ValueError(f'{pair_run}: domain/evaluation/row-count mismatch')
            if report.get('split', split) not in (split, 'development' if split == 'dev' else split):
                raise ValueError(f'{pair_run}: wrong evaluation split')
            g, v = short(report['g']), short(report['v'])
            if g == v:
                raise ValueError(f'{pair_run}: identical model roles')
            runs = report['runs']
            if len(runs) != (4 if group['kind'] == 'selector' and split == 'test' else 2):
                raise ValueError(f'{pair_run}: unexpected upstream runs')
            upstream = [ar.root / Path(name).name for name in runs[-2:]]
            outcomes = []
            for model, run in zip((g, v), upstream):
                cfg = ar.config(run)
                ds = cfg['dataset']
                key = (Path(ds['path']).parent.name, ds['manifest'])
                entry = entries.get(key)
                if short(cfg['model']['model_id']) != model or not entry or (entry['domain'], entry['split']) != (domain, split):
                    raise ValueError(f'{pair_run}: upstream model/dataset/split mismatch')
                grades = ar.grades(run, 'u0')
                if len(grades) != n:
                    raise ValueError(f'{pair_run}: upstream u0 row-count mismatch')
                # Reports with historical upstream hashes must agree with the current archive.
                prior = {h for path, h in report.get('source_sha256', {}).items()
                         if Path(path).parent.parent.name == run.name and Path(path).name == 'grades.jsonl'}
                if prior and prior != {ar.sources[str(run / 'records/grades.jsonl')]}:
                    raise ValueError(f'{pair_run}: upstream grade hash differs from archived report')
                outcomes.append(grades)
            ug, uv = outcomes
            dataset_id = key_dataset(upstream, ar)
            if set(ug) != set(uv):
                raise ValueError(f'{pair_run}: solo u0 memberships differ')
            solo_counts = {'G': sum(ug.values()), 'V': sum(uv.values()),
                           'oracle': sum(ug[p] or uv[p] for p in ug)}
            if any(report[k] != c for k, c in solo_counts.items()):
                raise ValueError(f'{pair_run}: archived solo/oracle summary differs from u0')
            design = any(set(p['models']) == {g, v} for p in labels['domains'][domain]['design_pairs'])
            models = sorted((g, v))
            systems = ('primary', 'secondary') if group['kind'] == 'selector' else ('JEV',)
            if group['kind'] not in ('selector', 'JEV'):
                raise ValueError('unknown system source kind')
            if group['kind'] == 'JEV' and any(report.get(k, 0) != 0 for k in
                    ('failed_rows', 'failed_calls', 'missing_rows', 'unparseable_judge_rows')):
                raise ValueError(f'{pair_run}: archived JEV has failed/missing/unparseable rows')
            for kind in systems:
                system = 'selector_' + kind if kind != 'JEV' else kind
                key = (domain, split, tuple(models), system)
                if key in seen:
                    raise ValueError(f'{pair_run}: duplicate pair/system source')
                seen.add(key)
                path = pair_run / 'records' / ('decisions.jsonl' if kind == 'JEV' else f'{kind}_decisions.jsonl')
                decisions = [json.loads(line) for line in checked(path).read_text(encoding='utf-8').splitlines()]
                ids = [str(r['problem_id']) for r in decisions]
                if len(ids) != n or len(set(ids)) != n or set(ids) != set(ug):
                    raise ValueError(f'{pair_run}: missing/duplicate/extra decision rows')
                seeded = report['selectors'][kind].get('per_seed') if kind != 'JEV' else None
                seed_counts = [0] * len(seeded) if seeded is not None else None
                total = 0
                for r in decisions:
                    pid = str(r['problem_id'])
                    if type(r['pick_v']) is not bool:
                        raise ValueError(f'{pair_run}: non-boolean choice')
                    if kind == 'JEV' and r.get('prob_v') is not None:
                        p = r['prob_v']
                        if type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 or r['pick_v'] != (p >= .5):
                            raise ValueError(f'{pair_run}: JEV choice/probability mismatch')
                    if kind != 'JEV':
                        probs = [r] + r.get('per_seed', [])
                        if any(type(p.get(k)) not in (int, float) or not math.isfinite(p[k]) or not 0 <= p[k] <= 1
                               for p in probs for k in ('p_g', 'p_v')):
                            raise ValueError(f'{pair_run}: invalid selector probability')
                        if r['pick_v'] != (r['p_v'] >= r['p_g']):
                            raise ValueError(f'{pair_run}: choice differs from selector probabilities')
                        if seeded is not None:
                            if len(r.get('per_seed', [])) != len(seeded):
                                raise ValueError(f'{pair_run}: missing per-seed decisions')
                            if [p['seed'] for p in r['per_seed']] != report['seeds']:
                                raise ValueError(f'{pair_run}: different seed order')
                            for j, p in enumerate(r['per_seed']):
                                seed_counts[j] += uv[pid] if p['p_v'] >= p['p_g'] else ug[pid]
                            for k in ('p_g', 'p_v'):
                                if not math.isclose(r[k], sum(p[k] for p in r['per_seed']) / len(seeded), abs_tol=1e-15, rel_tol=0):
                                    raise ValueError(f'{pair_run}: average probability differs from seeds')
                    correct = uv[pid] if r['pick_v'] else ug[pid]
                    total += correct
                    rows.append(dict(domain=domain, dataset_id=dataset_id, split=split,
                                     evaluation=group['evaluation'], system=system, pair_id='|'.join(models),
                                     g_model=g, v_model=v, problem_id=pid, correct=correct,
                                     in_design_sample=design, run_id=str(pair_run.relative_to(root)),
                                     source_run=root.name, upstream_runs=[r.name for r in upstream]))
                expected = report['JEV'] if kind == 'JEV' else report['selectors'][kind]['correct']
                if total != expected or (seeded is not None and seed_counts != seeded):
                    raise ValueError(f'{pair_run}: recomputed {system} accuracy/per-seed counts differ from archive')
                summaries.append(dict(domain=domain, split=split, evaluation=group['evaluation'], system=system,
                                      models=models, g_model=g, v_model=v, n=n, correct=total,
                                      archived_correct=expected, per_seed=seed_counts, in_design_sample=design,
                                      upstream_runs=[r.name for r in upstream], source=str(pair_run), **solo_counts))
            used_files.update(str(p.relative_to(root)).replace('\\', '/') for p in
                              [report_path, pair_run / 'reports/summary.md'] +
                              [pair_run / 'records' / ('decisions.jsonl' if k == 'JEV' else f'{k}_decisions.jsonl') for k in systems])
        if used_files != set(hashes):
            raise ValueError(f'{root.name}: registry file membership differs from discovered sources')
    return rows, summaries


def key_dataset(upstream: list[Path], ar: Archives) -> str:
    return Path(ar.config(upstream[0])['dataset']['path']).parent.name


def evaluation_set(domain: str, split: str) -> str:
    # 'cv' names the comparison set, not a claim that a solo/review/JEV was fitted.
    return 'cv' if domain in ('code', 'nl_reasoning') else split


def build(artifacts: Path, labels: dict, system_sources: dict | None = None) -> dict:
    ar = Archives(artifacts)
    found, ignored = discover(artifacts, labels, ar)
    prefer = set(labels.get("prefer", []))
    solos_all = [f for f in found if f["kind"] == "solo"]
    by_manifest, amb_manifest = choose(_group(solos_all, lambda f: (f["dataset"], f["manifest"], f["model"])), prefer)
    table_groups = _group([f for f in solos_all if f["entry"].get("solo_table", True)],
                          lambda f: (f["domain"], f["split"], f["model"]))
    solo_table, amb_solo = choose(table_groups, prefer)
    dir_groups = _group([f for f in found if f["kind"] == "direction"],
                        lambda f: (f["domain"], f["split"], f["model"], f["generator"]))
    directions, amb_dir = choose(dir_groups, prefer)

    solo_rows, solo_ids = [], defaultdict(set)
    counts = defaultdict(lambda: defaultdict(dict))
    dir_rows, dir_index, problems = [], defaultdict(lambda: defaultdict(dict)), []
    bad_runs = set()
    for (domain, split, model), f in sorted(solo_table.items()):
        grades, problem = read_grades(ar, f["run"], "u0")
        if problem:
            problems.append(problem)
            bad_runs.add(f["run"])
            continue
        counts[domain][model][split] = {"correct": sum(grades.values()), "n": len(grades)}
        solo_ids[(domain, split)] |= set(grades)
        solo_rows += [{"domain": domain, "split": split, "model": model, "problem_id": pid,
                       "correct": u, "condition": f["condition"],
                       "evaluation": evaluation_set(domain, split)} for pid, u in sorted(grades.items())]

    for (domain, split, reviewer, generator), f in sorted(directions.items()):
        g_run = f["g_archive"]
        v_src = by_manifest.get((f["dataset"], f["manifest"], reviewer)) or solo_table.get((domain, split, reviewer))
        if v_src and v_src["run"] in bad_runs:
            v_src = None
        (yg, pg), (yy, py) = read_grades(ar, g_run, "u0"), read_grades(ar, f["run"], "u_gv")
        yv, pv = read_grades(ar, v_src["run"], "u0") if v_src else ({}, None)
        if pg or py:
            problems += [f"{f['condition']}: " + x for x in (pg, py) if x]
            continue
        if pv:
            problems.append(f"{f['condition']}: V solo unusable, v left empty: {pv}")
        ids = set(yy)
        if f["entry"].get("restrict_to"):
            ids &= solo_ids[(domain, f["entry"]["restrict_to"])]
        if not ids <= set(yg):
            problems.append(f"{f['condition']}: reviewed rows missing from G source")
            continue
        dir_index[domain][(generator, reviewer)][split] = f["condition"]
        dir_rows += [{"domain": domain, "split": split, "generator": generator, "reviewer": reviewer,
                      "problem_id": pid, "g": yg[pid], "v": yv.get(pid), "gv": yy[pid],
                      "condition": f["condition"], "v_condition": v_src["condition"] if v_src else None,
                      "evaluation": evaluation_set(domain, split)}
                     for pid in sorted(ids)]

    system_rows, system_summaries = ingest_systems(ar, labels, system_sources) if system_sources else ([], [])
    system_pairs = defaultdict(set)
    for r in system_summaries:
        system_pairs[r['domain']].add(tuple(r['models']))
    pairs, missing = {}, []
    for domain, spec in labels["domains"].items():
        tiers = {m: t for t, ms in (spec.get("tiers") or {}).items() for m in ms}
        design = {frozenset(p["models"]): p for p in spec.get("design_pairs", [])}
        split_for_strength = labels["stronger_from"][domain]
        models = {m for g, v in dir_index[domain] for m in (g, v)}
        models |= {m for pair in system_pairs[domain] for m in pair}
        out = []
        for a, b in itertools.combinations(sorted(models), 2):
            dirs = {"G=" + g: dir_index[domain].get((g, r), {}) for g, r in ((a, b), (b, a))}
            if not any(dirs.values()) and (a, b) not in system_pairs[domain]:
                continue
            ca = counts[domain].get(a, {}).get(split_for_strength, {}).get("correct")
            cb = counts[domain].get(b, {}).get(split_for_strength, {}).get("correct")
            stronger = None if ca is None or cb is None or ca == cb else (a if ca > cb else b)
            d = design.get(frozenset((a, b)))
            out.append({"models": [a, b], "tiers": [tiers.get(a), tiers.get(b)], "stronger": stronger,
                        "stronger_from": {a: ca, b: cb, "split": split_for_strength},
                        "in_design_sample": d is not None, "cell": d and d["cell"],
                        "headroom": d and d["headroom"], "directions": dirs})
        for key, d in design.items():
            have = next((p for p in out if frozenset(p["models"]) == key), None)
            if have is None or not all(have["directions"].values()):
                missing.append(f"{domain} {d['cell']} {d['headroom']}: {sorted(key)}")
        pairs[domain] = out
    return {"solo_rows": solo_rows, "dir_rows": dir_rows, "pairs": pairs, "counts": counts,
            "missing_design": missing, "ambiguous": amb_solo + amb_dir + amb_manifest, "ignored": ignored,
            "problems": problems, "sources": ar.sources,
            "system_rows": system_rows, "system_summaries": system_summaries}


def _group(items, key):
    groups = defaultdict(list)
    for item in items:
        groups[key(item)].append(item)
    return groups


def write(result: dict, out: Path) -> None:
    (out / "db").mkdir(parents=True)
    (out / "reports").mkdir()
    for name in ("solo", "dir", "system"):
        rows = result[f"{name}_rows"]
        path = out / "db" / ({"solo": "solo.jsonl", "dir": "directions.jsonl", "system": "systems.jsonl"}[name])
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (out / "db/pairs.json").write_text(json.dumps(result["pairs"], ensure_ascii=False, indent=2) + "\n")
    (out / "reports/inventory.json").write_text(json.dumps(
        {"source_sha256": result["sources"],
         "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2) + "\n")
    (out / 'reports/systems.json').write_text(json.dumps(
        {'metric': 'u0', 'all_archived_counts_match': True, 'systems': result['system_summaries']},
        ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    lines = ["# 结果库清单", "",
             "逐题数据在 `db/`，只留在服务器。本页只有计数，可以提交。", "",
             "|域|配对数|设计配对|两方向都有的设计配对|方向（各切分合计）|",
             "|---|---:|---:|---:|---:|"]
    for domain, ps in result["pairs"].items():
        design = [p for p in ps if p["in_design_sample"]]
        lines.append(f"|{domain}|{len(ps)}|{len(design)}|{sum(all(p['directions'].values()) for p in design)}|"
                     f"{sum(len(s) for p in ps for s in p['directions'].values())}|")
    lines += ["", "## 单独作答（正确/题数）", "", "|域|模型|各切分|", "|---|---|---|"]
    for domain, models in result["counts"].items():
        for model, splits in sorted(models.items()):
            lines.append(f"|{domain}|{model}|" + "；".join(f"{s} {c['correct']}/{c['n']}" for s, c in sorted(splits.items())) + "|")
    lines += ["", "## 较强方无法确定的配对", ""]
    lines += [f"- {d}: {p['models']}" for d, ps in result["pairs"].items() for p in ps if p["stronger"] is None] or ["- 无"]
    for title, key in (("缺少的设计配对方向", "missing_design"), ("有歧义、未纳入的归档（需在 labels.yaml 的 prefer 里指定）", "ambiguous"),
                       ("读取问题", "problems"), ("忽略的归档", "ignored")):
        lines += ["", f"## {title}", ""] + ([f"- {x}" for x in result[key]] or ["- 无"])
    lines += ['', '## 选择器与 JEV：逐对与归档一致', '',
              '每行保留 split（dev/test）及 evaluation（dev/test/cv）。代码、推理的 cv 标识共同比较题集；只有选择器经过交叉拟合。',
              'H01–H08 的卡片与题集保持不变；RQ3/RQ4 将另行筛选数学、知识的 test1500。', '',
              '|域|配对|题集|system|正确/题数|设计配对|', '|---|---|---|---|---:|---|']
    for r in result['system_summaries']:
        lines.append(f"|{r['domain']}|{' / '.join(r['models'])}|{r['evaluation']}|{r['system']}|{r['correct']}/{r['n']}|{r['in_design_sample']}|")
    (out / "reports/summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    ap.add_argument("--labels", type=Path, default=ROOT / "claims/labels.yaml")
    ap.add_argument('--system-sources', type=Path, default=ROOT / 'claims/system_sources.json')
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    if args.out_dir.exists():
        raise SystemExit(f"{args.out_dir} exists; results databases are never overwritten")
    try:
        result = build(args.artifacts, yaml.safe_load(args.labels.read_text()),
                       json.loads(args.system_sources.read_text(encoding='utf-8')))
        for gate in ('missing_design', 'ambiguous', 'problems'):
            if result[gate]:
                raise ValueError(f"results database {gate} gate failed: {result[gate]}")
    except Exception as exc:
        (args.out_dir / 'reports').mkdir(parents=True)
        (args.out_dir / 'reports/summary.md').write_text(
            f'# 结果库检查失败\n\n{type(exc).__name__}: {exc}\n\n没有写逐题库，没有放宽检查。\n', encoding='utf-8')
        raise
    result['sources'][str(args.labels)] = hashlib.sha256(args.labels.read_bytes()).hexdigest()
    result['sources'][str(args.system_sources)] = hashlib.sha256(args.system_sources.read_bytes()).hexdigest()
    write(result, args.out_dir)
    print(json.dumps({"solo_rows": len(result["solo_rows"]), "direction_rows": len(result["dir_rows"]),
                      'system_rows': len(result['system_rows']), 'system_pairs': len(result['system_summaries']),
                      "missing_design": len(result["missing_design"]), "ambiguous": len(result["ambiguous"]),
                      "problems": len(result["problems"])}))


if __name__ == "__main__":
    main()
