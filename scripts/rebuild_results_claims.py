#!/usr/bin/env python3
"""Rerun unchanged H01-H08 cards and figures, using an already validated new database.

Only aggregate reports are copied into the delivery directory. This performs no
generation, grading, new hypothesis selection or cost calculation.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN = '主分析：设计配对'


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def comparison(out, baseline):
    comparisons = []
    labels = {'H01': '较强者生成方向 − 较强者审阅方向', 'H02': '审阅后 − 被审者',
              'H03': '审阅后 − 较强者单独', 'H04': '两者都错时修复 − 两者都对时破坏',
              'H05': '一对一错时，跟随被审者 − 跟随审阅者', 'H06': '一对一错比例：独立作答 − 审阅后',
              'H07': '两独立答案的逐题上限 − 较强者', 'H08': '能力差距与上限空间的域内相关系数'}
    lines = ['# H01–H08 更新（与 20261004 解读采用的报告比较）', '',
             '卡片、题集、2000 次 bootstrap 与种子均不变。数学、知识 2500 题；代码 832 题；推理 901 题。',
             'H01 在 20261004 目录没有独立报告，该版本引用的 20261003b/H01 用作对照。',
             '选择器/JEV 不参与 H01–H08；数学、知识的新 system 只收 test1500，代码、推理收 cv。', '',
             '|假说|主要量（H01–H07 为 pp）|旧值|新值（95% 区间）|变化|', '|---|---|---:|---|---:|']
    for number in range(1, 9):
        h = f'H{number:02d}'
        prior = baseline / f'{h}_result.json'
        if h == 'H01' and not prior.exists():
            prior = baseline.parent / '20261003b/H01_result.json'
        new_path = out / f'{h}_result.json'
        old, new = load(prior)['analyses'][MAIN], load(new_path)['analyses'][MAIN]
        if h == 'H08':
            a, b = old['correlation']['statistics']['合计'], new['correlation']['statistics']['合计']
            old_value, value, ci = a['point'], b['point'], b['ci95']
            per_domain = None
            extras = {}
            for kind in ('stronger_reviews', 'weaker_reviews'):
                extras[kind] = dict(old_categories=dict(Counter(d['category'] for d in old['directions'] if d['kind'] == kind)),
                                    categories=dict(Counter(d['category'] for d in new['directions'] if d['kind'] == kind)),
                                    joint_slopes={k: {'old': old['fits'][kind]['statistics'][k], 'new': v}
                                                 for k, v in new['fits'][kind]['statistics'].items() if k.startswith('合计|dC|both|')})
        else:
            a, b = old['report']['bootstrap'], new['report']['bootstrap']
            old_value, value, ci = a['overall']['mean_diff_pp'], b['overall']['mean_diff_pp'], b['overall']['ci95_pp']
            per_domain = {d: {'old': a['per_domain'][d], 'new': v,
                             'change_pp': v['mean_diff_pp'] - a['per_domain'][d]['mean_diff_pp']}
                          for d, v in b['per_domain'].items()}
            extras = {}
        delta = value - old_value
        comparisons.append(dict(claim=h, quantity=labels[h], baseline=str(prior),
                                baseline_sha256=hashlib.sha256(prior.read_bytes()).hexdigest(),
                                new_sha256=hashlib.sha256(new_path.read_bytes()).hexdigest(),
                                old=old_value, new=value, ci95=ci, change=delta, per_domain=per_domain, **extras))
        lines.append(f'|{h}|{labels[h]}|{old_value:+.3f}|{value:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]|{delta:+.3f}|')
    lines += ['', '## 逐域变化（pp）', '', '|假说|域|旧值|新值|变化|新 95% 区间|', '|---|---|---:|---:|---:|---|']
    for c in comparisons:
        for domain, v in (c['per_domain'] or {}).items():
            a, b = v['old'], v['new']
            lo, hi = b['ci95_pp']
            lines.append(f"|{c['claim']}|{domain}|{a['mean_diff_pp']:+.3f}|{b['mean_diff_pp']:+.3f}|{v['change_pp']:+.3f}|[{lo:+.3f}, {hi:+.3f}]|")
    dirs = load(out / 'H08_result.json')['analyses'][MAIN]['directions']
    counts = dict(Counter(d['domain'] for d in dirs))
    if counts != {d: 24 for d in ('math', 'knowledge', 'code', 'nl_reasoning')}:
        raise ValueError(f'expected 24 design review directions per domain; found {counts}')
    lines += ['', '四域各 24 个设计方向，共 96 个；推理域从 13 个补齐到 24 个。其余配对仅在敏感性分析中使用。',
              '没有新增 RQ3/RQ4 假说，没有计算成本。', '',
              'H09（事后、探索性）报告：H09_summary.md。图在本地用 plot_rq1.py、plot_rq2.py 从提交的 JSON 画。']
    (out / 'comparison.json').write_text(json.dumps(comparisons, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (out / 'reports/summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--db', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--baseline', type=Path, default=ROOT / 'docs/analysis/claims/20261004')
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit('completed/attempted reports are immutable; choose a new output directory')
    args.out.mkdir(parents=True)
    (args.out / 'reports').mkdir()
    try:
        if not load(args.db / 'reports/systems.json')['all_archived_counts_match']:
            raise ValueError('system accuracy validation did not pass')
        for i in range(1, 9):
            h = f'H{i:02d}'
            cards = list((ROOT / 'claims').glob(f'{h}_*.yaml'))
            if len(cards) != 1:
                raise ValueError(f'{h}: expected one registered card')
            script = 'explain_conditions.py' if h == 'H08' else 'evaluate_claim.py'
            target = args.out / 'runs' / h
            subprocess.run([sys.executable, str(ROOT / 'scripts' / script), '--db', str(args.db),
                            '--claim', str(cards[0]), '--out-dir', str(target)], check=True)
            for name, dest in [('result.json', f'{h}_result.json'), ('summary.md', f'{h}_summary.md')]:
                shutil.copyfile(target / 'reports' / name, args.out / dest)
        subprocess.run([sys.executable, str(ROOT / 'scripts/rq2_cells.py'), str(args.out)], check=True)
        # H09 reads only H08's per-direction summary. Figures are drawn locally from the committed
        # JSON (scripts/plot_rq1.py, scripts/plot_rq2.py); the cluster environment has no matplotlib.
        subprocess.run([sys.executable, str(ROOT / 'scripts/explain_outcome_levels.py'), '--h08',
                        str(args.out / 'H08_result.json'), '--out-dir', str(args.out)], check=True)
        comparison(args.out, args.baseline)
    except Exception as exc:
        (args.out / 'reports/summary.md').write_text(
            f'# H01–H08 重跑失败\n\n{type(exc).__name__}: {exc}\n\n未放宽检查。\n', encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
