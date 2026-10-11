#!/usr/bin/env python3
"""Server-only, deterministic development examples for the example figure.

Export only the first eight candidates per category without problem IDs. Full ID audit
stays in records/ on the server. No inference, regrading, or test problem reads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from gv_domain_tables import Archives
import run_c002_dev30 as base

DATASET = 'mathfull-r465bcdb-l3to5-s20260924'
MANIFEST_HASH = '52c6c79d0285f40d3ac711ad6331b8102d6cf138374a8fd93a5cdc6b3756971a'


def rows(path):
    with path.open(encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)


def indexed(path):
    out = {}
    for row in rows(path):
        pid = str(row['problem_id'])
        if pid in out:
            raise ValueError('Duplicate record')
        out[pid] = row
    return out


def output(row):
    out = row['outputs'][0]
    if row['y0_sample_index'] != 0 or out['sample_index'] != 0:
        raise ValueError('Not sample_index=0')
    if not isinstance(out['raw_output'], str):
        raise ValueError('Missing output text')
    return out


def length_key(candidate):
    return candidate['total_chars'], candidate['problem_id']


def select_shortest(candidates):
    if len(candidates) < 8:
        raise ValueError('Fewer than eight eligible candidates; do not relax criteria')
    return sorted(candidates, key=length_key)[:8]


def has_answer(out):
    # A wrong/unparseable grade is not itself an identifiable wrong answer.
    return base.answer_text(out) is not None


def gold_for_selected(selected):
    # Read only the source's ID column before selecting the authorized rows. Unlike
    # load_mathfull_rows, never iterate source problems outside the selected dev rows.
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    os.environ['HF_HUB_OFFLINE'] = '1'
    from datasets import load_dataset
    ds = load_dataset('nlile/hendrycks-MATH-benchmark',
                      revision='465bcdb36f5962aa3512891498966df785fc3c18', split='train')
    wanted = {r['problem_id'] for r in selected}
    indices = [i for i, pid in enumerate(ds.select_columns(['unique_id'])['unique_id'])
               if str(pid) in wanted]
    if len(indices) != len(wanted):
        raise ValueError('Selected IDs missing or duplicated in pinned source')
    return {str(r['unique_id']): r for r in
            ds.select(indices).select_columns(['unique_id', 'problem', 'answer'])}


def block(text):
    # Preserve every character; fence long enough for any archived content.
    fence = '`' * max(3, max((len(s) for s in text.split() if set(s) == {'`'}), default=0) + 1)
    return f'{fence}text\n{text}\n{fence}\n'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--artifacts', type=Path, required=True)
    ap.add_argument('--results-db', type=Path, required=True)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    reports = args.out_dir / 'reports'
    records = args.out_dir / 'records'
    reports.mkdir(); records.mkdir()
    (reports / 'summary.md').write_text('# Example selection\n\nStarted; not yet validated.\n', encoding='utf-8')
    try:
        run(args, reports, records)
    except BaseException as exc:
        # Error messages deliberately omit IDs and problem text.
        with (reports / 'summary.md').open('a', encoding='utf-8') as f:
            f.write(f'\nFAILED: {type(exc).__name__}. No completed export. Inspect server job log.\n')
        raise


def run(args, reports, records):
    if args.manifest.name != 'testpool_dev1000.jsonl' or args.manifest.parent.name != DATASET:
        raise ValueError('Only the registered development manifest is allowed')
    if hashlib.sha256(args.manifest.read_bytes()).hexdigest() != MANIFEST_HASH:
        raise ValueError('Development manifest hash mismatch')
    ids = {str(r['problem_id']) for r in rows(args.manifest)}
    if len(ids) != 1000:
        raise ValueError('Not 1000 unique development rows')
    ar = Archives(args.artifacts)
    solo = {cid: ar.find(cid) for cid in ('c201', 'c207')}
    for cid, path in solo.items():
        if path is None or ar.config(path)['dataset']['manifest'] != 'testpool_dev1000':
            raise ValueError('Solo archive missing or not development')
    dbfile = args.results_db / 'db/directions.jsonl'
    db = {cid: {} for cid in ('c216', 'c217')}
    for row in rows(dbfile):
        cid = row['condition']
        if cid in db:
            if row['domain'] != 'math' or row['split'] != 'dev' or row['evaluation'] != 'dev':
                raise ValueError('Unexpected results DB split')
            pid = str(row['problem_id'])
            if pid in db[cid]:
                raise ValueError('Duplicate results DB row')
            db[cid][pid] = row
    audit, chosen, summaries = [], [], {}
    for category, cid, g_cid, v_cid in [('A', 'c216', 'c207', 'c201'), ('B', 'c217', 'c201', 'c207')]:
        review = ar.find(cid)
        if review is None:
            raise ValueError('Review archive missing')
        cfg = ar.config(review)
        if cfg['dataset']['manifest'] != 'testpool_dev1000':
            raise ValueError('Review is not development')
        g_run = args.artifacts / Path(cfg['generator_source']['artifact_path']).name
        if g_run != solo[g_cid] or cfg['generator_source']['condition_id'] != g_cid:
            raise ValueError('Unexpected actual generator source')
        g = indexed(ar.checked(g_run, 'records/pass1.jsonl'))
        copied = indexed(ar.checked(review, 'records/g_pass1.jsonl'))
        v = indexed(ar.checked(solo[v_cid], 'records/pass1.jsonl'))
        rev = indexed(ar.checked(review, 'records/review.jsonl'))
        yg, yv, yr = ar.grades(g_run), ar.grades(solo[v_cid]), ar.grades(review, 'u_gv')
        if any(set(t) != ids for t in (g, copied, v, rev, yg, yv, yr, db[cid])):
            raise ValueError('Missing or extra development rows')
        eligible, event_count, skipped_asy, fallback_count, no_g_answer = [], 0, 0, 0, 0
        for pid in sorted(ids):
            if copied[pid] != g[pid]:
                raise ValueError('Generator copy differs from actual source')
            go, vo, ro = output(g[pid]), output(v[pid]), output(rev[pid])
            if rev[pid]['source_g_output_sha256'] != hashlib.sha256(go['raw_output'].encode()).hexdigest():
                raise ValueError('Reviewer saw a different generator output')
            if rev[pid]['source_g_record_id'] != g[pid]['record_id'] or rev[pid]['source_g_run_id'] != g[pid]['run_id']:
                raise ValueError('Review source identity mismatch')
            problem = rev[pid]['problem_text']
            if problem != g[pid]['problem_text'] or problem != v[pid]['problem_text']:
                raise ValueError('Problem text mismatch')
            d = db[cid][pid]
            if (d['g'], d['v'], d['gv'], d['v_condition']) != (yg[pid], yv[pid], yr[pid], v_cid):
                raise ValueError('Archived outcomes do not match results DB')
            target = (0, 1, 0) if category == 'A' else (1, 0, 0)
            if (yg[pid], yv[pid], yr[pid]) != target:
                continue
            event_count += 1
            if '[asy]' in problem:
                skipped_asy += 1
                continue
            if not has_answer(ro):
                fallback_count += 1
                continue
            eq = None
            if category == 'A':
                if not has_answer(go):
                    no_g_answer += 1
                    continue
                eq, left_ok, right_ok = base.math_equivalent(base.answer_text(go), base.answer_text(ro))
                if eq is None or not left_ok or not right_ok:
                    raise ValueError('Answer equivalence could not be parsed')
                if not eq:
                    continue
            lens = [len(problem), len(go['raw_output']), len(ro['raw_output'])]
            eligible.append(dict(category=category, problem_id=pid, problem=problem,
                                 g_output=go['raw_output'], g_answer=go['extracted_answer'],
                                 v_solo_tail=vo['raw_output'][-400:], v_solo_answer=vo['extracted_answer'],
                                 review_output=ro['raw_output'], final_answer=ro['extracted_answer'],
                                 chars=lens, total_chars=sum(lens), equivalent_to_g=eq))
        selected = select_shortest(eligible)
        chosen.extend(selected)
        audit.extend({'category': category, 'rank': i + 1, 'problem_id': r['problem_id'],
                      'total_chars': r['total_chars'], 'selected': i < 8}
                     for i, r in enumerate(sorted(eligible, key=length_key)))
        summaries[category] = dict(review_run=str(review), actual_g_source=str(g_run),
                                  v_solo_source=str(solo[v_cid]), rows_checked=1000,
                                  outcome_cell=event_count, skipped_asy=skipped_asy,
                                  review_fallbacks_skipped=fallback_count, eligible=len(eligible),
                                  no_identifiable_g_answer_skipped=no_g_answer,
                                  selected=len(selected), selected_lengths=[r['total_chars'] for r in selected])
    gold = gold_for_selected(chosen)
    lines = ['# 真实例子候选', '', '每类按总字符数从短到长取前 8；A/B 编号是候选顺序，不是题号。', '']
    for category in ('A', 'B'):
        lines.extend([f'## {category} 类：' + ('Qwen3-8B 采纳 Llama-3.1-8B 的错答案' if category == 'A' else 'Llama-3.1-8B 把 Qwen3-8B 的正确答案改错'), ''])
        for i, r in enumerate(x for x in chosen if x['category'] == category):
            source = gold[r['problem_id']]
            if source['problem'] != r['problem']:
                raise ValueError('Selected problem differs from pinned source')
            label = f'{category}{i + 1:02}'
            lines.extend([f'### {label}', '',
                          f"字符数：题目 {r['chars'][0]}，G 输出 {r['chars'][1]}，审阅输出 {r['chars'][2]}，总计 {r['total_chars']}。", ''])
            for title, text in [('题目全文', r['problem']), ('标准答案', source['answer']),
                                ('G 输出全文', r['g_output']), ('G 抽出的答案', str(r['g_answer'])),
                                ('V 单独作答抽出的答案', str(r['v_solo_answer'])),
                                ('V 单独作答输出最后 400 字', r['v_solo_tail']),
                                ('V 审阅输出全文', r['review_output']), ('审阅后抽出的最终答案', str(r['final_answer']))]:
                lines.extend([f'**{title}**', '', block(text)])
    candidate_text = '\n'.join(lines)
    if any(pid in candidate_text for pid in ids):
        raise ValueError('A problem ID leaked into candidate text')
    commit = (ROOT / 'SOURCE_COMMIT.txt').read_text().strip() if (ROOT / 'SOURCE_COMMIT.txt').exists() else subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    summary = dict(source_commit=commit, results_db=str(args.results_db), manifest_sha256=MANIFEST_HASH,
                   categories=summaries, sources=ar.sources, missing_rows=0, equivalence_errors=0,
                   sorting='len(problem)+len(actual G output)+len(raw review output); ties by server-only problem_id',
                   test_problem_texts_read=0, source_gold_rows_read=len(gold),
                   candidate_sha256=hashlib.sha256(candidate_text.encode()).hexdigest())
    (records / 'selection_audit.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in audit), encoding='utf-8')
    (reports / 'selection.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (reports / 'candidates.md').write_text(candidate_text, encoding='utf-8')
    readme = f"# 数学开发集真实例子候选\n\n执行提交：`{commit}`。结果库：`{args.results_db}`。\n\n仅使用 `testpool_dev1000`，清单 SHA256：`{MANIFEST_HASH}`。\n\n"
    for cat, s in summaries.items():
        readme += f"- {cat} 类：符合条件 {s['eligible']} 道，取前 {s['selected']} 道；对错格共 {s['outcome_cell']} 道，跳过含图题 {s['skipped_asy']} 道、审阅无答案回退 {s['review_fallbacks_skipped']} 道、G 无可识别答案 {s['no_identifiable_g_answer_skipped']} 道。\n  审阅：`{s['review_run']}`。实际 G 来源：`{s['actual_g_source']}`。V 独立作答：`{s['v_solo_source']}`。\n"
    readme += '\n排序：题目字符数 + 实际 G 输出字符数 + V 审阅输出字符数；同长度按服务器题号排序，未手挑。V 独立输出尾部不计入排序长度。\n\n全部 2,000 个方向行与结果库和归档成绩逐行一致，实际 G 副本与来源逐行一致；来源校验和通过。对错采用既有 u0 / u_gv，未重新判分。A 类等价检查使用现有 math_equivalent 对原始输出解析，不重新解析 SymPy 字符串。无答案的回退不作为主动采纳例子。\n\n标准答案仅从固定公开数据源读取已选开发题；先读取 ID 列定位，再读取所选行的题目与答案，未读取测试题目/输出。完整排序和题号审计只留服务器 records/selection_audit.jsonl。候选按原文导出，选定后再画图。\n'
    (reports / 'README.md').write_text(readme, encoding='utf-8')
    (reports / 'summary.md').write_text(readme + '\n状态：完成，缺行 0，等价检查错误 0。\n', encoding='utf-8')
    print(json.dumps({'status': 'complete', 'categories': summaries}, ensure_ascii=False))


if __name__ == '__main__':
    main()
