#!/usr/bin/env python3
"""Frozen knowledge selector, nested OOF on the corrected twelve NL pairs."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from authority_allocation.condition_resolution import resolve_condition
from authority_allocation import mc_pair_features as mpf
from gv_domain_tables import Archives, paired
from code_selector_backfill import assignments, validate_decisions
from code_jev_dev832 import read_rows, sample_zero, digest
import mc_pair_selector_dev1000 as mc
import selector_dev1000 as sel
from nl_jev_version_probe import REGISTRY, checked_file


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')


def checksums(out):
    (out/'checksums.sha256').write_text(''.join(
        f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(out).as_posix()}\n'
        for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'checksums.sha256'), encoding='utf-8')


def registration():
    reg = yaml.safe_load(REGISTRY.read_text(encoding='utf-8'))
    labels = yaml.safe_load((ROOT/'claims/labels.yaml').read_text(encoding='utf-8'))
    if [p['design'] for p in reg['pairs']] != labels['domains']['nl_reasoning']['design_pairs'] or len(reg['pairs']) != 12:
        raise ValueError('wrong design pairs')
    contract = yaml.safe_load((ROOT/'protocols/pair_selector_mc_oof_v1/features.yaml').read_text(encoding='utf-8'))
    if contract['feature_sets'] != mc.feature_sets() or contract['members'] != {
            k:[list(m) for m in v] for k,v in mc.MEMBERS.items()}:
        raise ValueError('knowledge features or ensembles changed')
    if contract['seeds'] != sel.SEEDS or (contract['outer_folds'],contract['inner_folds']) != (sel.OUTER,sel.INNER):
        raise ValueError('cross-fit settings changed')
    for fam in ('l2','gbt'):
        if contract['grids'][fam] != sel.search(fam, sel.SEEDS[0]).param_grid:
            raise ValueError('knowledge model grid changed')
    ds = yaml.safe_load((ROOT/f"datasets/{reg['dataset']}/dataset.yaml").read_text(encoding='utf-8'))
    for pair in reg['pairs']:
        cfg = resolve_condition(ROOT/f"conditions/{pair['selector_condition']}/condition.yaml", repository_root=ROOT)
        if cfg['pair'] != pair or cfg['controller']['feature_contract'] != 'protocols/pair_selector_mc_oof_v1/features.yaml':
            raise ValueError('wrong selector condition')
        if cfg['replicates']['plan_id'] != 'r017' or cfg['dataset']['manifest_sha256'] != ds['manifests']['eval']['sha256']:
            raise ValueError('wrong replicate/dataset')
        alias = yaml.safe_load((ROOT/cfg['dataset']['path']).read_text(encoding='utf-8'))
        if alias['manifests'] != {**ds['manifests'], 'eval':{
                **ds['manifests']['eval'], 'purpose':'development_crossfit_only_same_immutable_membership'}}:
            raise ValueError('development alias changes immutable membership')
        if alias['policies']['may_support_confirmatory_claims'] or not cfg['dataset']['development_only']:
            raise ValueError('development-only identity required')
    return reg, ds


def manifest_ids(ds):
    manifest = ds['manifests']['eval']
    path = checked_file(Path(manifest['server_path']), manifest['sha256'])
    ids = [str(r['problem_id']) for r in read_rows(path)]
    if len(ids) != 901 or len(set(ids)) != 901:
        raise ValueError('wrong manifest membership')
    return ids


def load_pair(reg, pair, ids, ar):
    loaded, runs, configs, letters = {}, {}, {}, {}
    for side in ('g','v'):
        cid = pair[side]['condition']
        frozen = reg['solos'][cid]
        run = ar.root/frozen['run']
        cfg = ar.config(run)
        checked_file(ar.checked(run,'condition.resolved.yaml'),frozen['sha256']['condition.resolved.yaml'])
        if cfg['condition_id'] != cid or cfg['model']['model_id'] != frozen['model']:
            raise ValueError('upstream model identity changed')
        if cfg['dataset']['id'] != reg['dataset']:
            raise ValueError('upstream dataset identity changed')
        records = {}
        for name in ('pass1','native_logit_trace','scalar_evidence'):
            rel = f'records/{name}.jsonl'
            path = checked_file(ar.checked(run, rel), frozen['sha256'][rel])
            rows = {}
            for row in read_rows(path):
                pid = str(row['problem_id'])
                if pid in rows:
                    raise ValueError('duplicate upstream row')
                rows[pid] = row
            if set(rows) != set(ids):
                raise ValueError('missing upstream evidence')
            records[name] = rows
        vocab_path = Path(cfg['model']['server_path'])/'tokenizer.json'
        vocab = json.loads(vocab_path.read_text(encoding='utf-8'))['model']['vocab']
        letters[side] = mpf.letter_map({tid:tok for tok,tid in vocab.items()})
        ar.sources[str(vocab_path)] = hashlib.sha256(vocab_path.read_bytes()).hexdigest()
        loaded[side], runs[side], configs[side] = records, run, cfg
    rows = {}
    for pid in ids:
        row, feats, extras = {}, {}, {}
        for side in ('g','v'):
            r = loaded[side]
            original = r['pass1'][pid]
            sample_zero(original)
            zero = next(o for o in original['outputs'] if o['sample_index'] == 0)
            # Strip all identifiers and run metadata from feature function inputs.
            record = {k:original[k] for k in ('problem_text','problem_metadata')}
            record['outputs'] = [zero]
            native = {'positions':r['native_logit_trace'][pid]['positions']}
            evidence = {k:r['scalar_evidence'][pid][k] for k in ('features','audit')}
            feats[side], extras[side] = mpf.model_features(record,native,evidence,letters[side])
            row.update({f'{side}_{k}':v for k,v in feats[side].items()})
        row.update(mpf.problem_features(record))
        row.update(mpf.cross_features(feats['g'],feats['v'],extras['g'],extras['v']))
        if set(row) != set(mc.feature_sets()['F3']['g']):
            raise ValueError('unexpected inference feature')
        rows[pid] = row
    return rows, runs, configs


def grades(reg, pair, ids, runs, ar):
    values = []
    for side in ('g','v'):
        cid = pair[side]['condition']
        checked_file(ar.checked(runs[side],'records/grades.jsonl'), reg['solos'][cid]['sha256']['records/grades.jsonl'])
        labels = ar.grades(runs[side], outcome='u0')
        if set(labels) != set(ids) or sum(labels.values()) != reg['solos'][cid]['correct']:
            raise ValueError('primary u0 labels differ from frozen upstream')
        values.append(np.asarray([labels[p] for p in ids],dtype=int))
    if values[0].sum() >= values[1].sum():
        raise ValueError('V must be the frozen stronger solo')
    return values


def condition_manifest(out, pair):
    cid = pair['selector_condition']
    cfg = resolve_condition(ROOT/f'conditions/{cid}/condition.yaml', repository_root=ROOT)
    referenced = ['protocols/pair_selector_mc_oof_v1/features.yaml',
        'protocols/pair_selector_mc_oof_v1/input.schema.json', 'replicates/r017.yaml',
        'src/authority_allocation/mc_pair_features.py', 'src/authority_allocation/text_mc_features.py',
        'scripts/mc_pair_selector_dev1000.py','scripts/selector_dev1000.py','scripts/nl_selector_crossfit.py',
        'pairs/nl_selector_jev_eval901.yaml',f"datasets/{cfg['dataset']['id']}/dataset.yaml"]
    hashes = {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in referenced}
    (out/'condition.resolved.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
    manifest = {'run_id':out.name,'condition_id':cid,'dataset_id':cfg['dataset']['id'],
                'replicate_plan_id':'r017','source_commit':(ROOT/'SOURCE_COMMIT.txt').read_text().strip(),
                'upstream_runs':cfg['upstream_runs'], 'referenced_file_sha256':hashes,
                'condition_sha256':digest({'behavior':cfg,'referenced_file_sha256':hashes})}
    dump(out/'RUN_MANIFEST.json',manifest)
    return manifest


def fit(index, root):
    reg, ds = registration()
    pair = reg['pairs'][index]
    out = root/f"nl_reasoning__{pair['id']}"
    out.mkdir(parents=True,exist_ok=False)
    (out/'records').mkdir(); (out/'reports').mkdir()
    try:
        cm = condition_manifest(out,pair)
        ids = manifest_ids(ds)
        ar = Archives(ROOT/'artifacts')
        rows,runs,configs = load_pair(reg,pair,ids,ar)
        # Complete unlabeled evidence precedes any grade access.
        with (out/'records/selector_evidence.jsonl').open('x',encoding='utf-8') as stream:
            for pid in ids:
                stream.write(json.dumps({'problem_id':pid,'features':rows[pid],
                                         'missingness':{k:v is None for k,v in rows[pid].items()}})+'\n')
        yg,yv = grades(reg,pair,ids,runs,ar)
        with (out/'records/training_targets.jsonl').open('x',encoding='utf-8') as stream:
            for i,pid in enumerate(ids):
                stream.write(json.dumps({'problem_id':pid,'u_g':int(yg[i]),'u_v':int(yv[i])})+'\n')
        folds = assignments(yg,yv)
        members = {}
        fsets = mc.feature_sets()
        for fs,fam in [('F2','l2'),('F3','gbt')]:
            xg = sel.matrix(rows,ids,fsets[fs]['g']); xv = sel.matrix(rows,ids,fsets[fs]['v'])
            predictions = []
            for seed in sel.SEEDS:
                pg,pv = sel.crossfit(xg,xv,yg,yv,fam,'A',seed)
                if not np.all(np.isfinite(pg)) or not np.all(np.isfinite(pv)):
                    raise ValueError('missing OOF probability')
                predictions.append((pg,pv))
                print(f"{pair['id']} {fs}-{fam} seed={seed} complete",flush=True)
            members[fs,fam] = predictions
        report = dict(domain='nl_reasoning',split='dev',evaluation='nested cross-fit; out-of-fold',
            pair=pair['id'],condition_id=pair['selector_condition'],g=pair['g']['model'],v=pair['v']['model'],
            g_condition=pair['g']['condition'],v_condition=pair['v']['condition'],
            runs=[runs[s].name for s in ('g','v')],n=901,G=int(yg.sum()),V=int(yv.sum()),
            oracle=int(np.maximum(yg,yv).sum()),metric='u0',manifest_sha256=ds['manifests']['eval']['sha256'],
            seeds=sel.SEEDS,outer_folds=sel.OUTER,inner_folds=sel.INNER,selectors={},
            feature_contract=mpf.FEATURE_CONTRACT,feature_sets=fsets,run_manifest=cm,
            missing_features={k:sum(rows[p][k] is None for p in ids) for k in fsets['F3']['g']},
            failed_rows=0,missing_rows=0)
        for name,configs in mc.MEMBERS.items():
            chosen = [(fs,fam) for fs,form,fam in configs]
            pgs = [np.mean([members[c][i][0] for c in chosen],axis=0) for i in range(5)]
            pvs = [np.mean([members[c][i][1] for c in chosen],axis=0) for i in range(5)]
            pg,pv = np.mean(pgs,axis=0),np.mean(pvs,axis=0)
            path = out/f'records/{name}_decisions.jsonl'
            with path.open('x',encoding='utf-8') as stream:
                for i,pid in enumerate(ids):
                    stream.write(json.dumps(dict(problem_id=pid,p_g=float(pg[i]),p_v=float(pv[i]),
                        pick_v=bool(pv[i]>=pg[i]),per_seed=[dict(seed=seed,fold=int(folds[j,i]),
                        p_g=float(pgs[j][i]),p_v=float(pvs[j][i])) for j,seed in enumerate(sel.SEEDS)]))+'\n')
            yy,seedcounts = validate_decisions(path,ids,yg,yv,folds)
            report['selectors'][name] = dict(correct=int(yy.sum()),per_seed=seedcounts,
                                             vs_baseline2=paired(yy.tolist(),yv.tolist()))
        report['source_sha256'] = ar.sources
        dump(out/'reports/selector_dev.json',report)
        (out/'reports/summary.md').write_text(f"# NL knowledge-selector OOF\n\n{pair['id']}: N=901; "
            f"G={report['G']}; V={report['V']}; primary={report['selectors']['primary']['correct']}; "
            f"secondary={report['selectors']['secondary']['correct']}.\n\n"
            "Knowledge features, models, preprocessing and grids unchanged. Five outer and five inner folds; "
            "all five seeds retained. Unique problem IDs ensure held-out rows are absent from each training fold. "
            "Development cross-fit evidence; no separate held-out NL test. Primary metric u0 only.\n",encoding='utf-8')
        checksums(out)
    except Exception as exc:
        (out/'reports/summary.md').write_text(f'# NL selector FAILED\n\n{type(exc).__name__}; stop without tuning.\n',encoding='utf-8')
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--index',type=int)
    ap.add_argument('--out-dir',type=Path)
    ap.add_argument('--validate-registration',action='store_true')
    args = ap.parse_args()
    if args.validate_registration:
        registration(); print('NL REGISTRATION PASSED: 12 pairs; unchanged knowledge feature/model contract'); return
    if not Path('/path/to/project').exists():
        raise SystemExit('server only')
    if args.index is None or not 0<=args.index<12 or args.out_dir is None:
        raise SystemExit('one of 12 indices and a fresh root required')
    fit(args.index,args.out_dir)


if __name__ == '__main__': main()
