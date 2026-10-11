#!/usr/bin/env python3
"""H09: which of five outcome levels each review direction reaches, by who reviews whom and the gap.

Reads only an H08 result JSON (per-direction Δ_G, Δ_C, their 95% intervals and the gap; no problem
IDs) and the H09 card, and writes H09_result.json and H09_summary.md into a new --out-dir. Runs
locally; needs no results database.

    python scripts/explain_outcome_levels.py \
        --h08 docs/analysis/claims/20261004b/H08_result.json \
        --card claims/H09_outcome_levels.yaml --out-dir docs/analysis/claims/20261004c
"""
import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from authority_allocation.outcome_levels import (  # noqa: E402
    LEVELS, LEVEL_NAMES, outcome_level, stratified_spearman)

KINDS = {'stronger_reviews': '较强者审较弱者', 'weaker_reviews': '较弱者审较强者'}
LEVEL_ZH = {1: '比 G 差', 2: '和 G 差不多', 3: '高于 G、低于较强者', 4: '和较强者差不多', 5: '超过较强者'}
DOMAINS = ['math', 'knowledge', 'code', 'nl_reasoning']
INDICATORS = {
    'short_of_stronger': lambda d: int(d['dC_ci95'][1] < 0),
    'worse_than_stronger': lambda d: int(outcome_level(d['dG_ci95'], d['dC_ci95']) == 1),
}
INDICATOR_ZH = {'short_of_stronger': '追不平较强者（Δ_C 区间整个在 0 以下）',
                'worse_than_stronger': '比较强者差（第 1 档）'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def band_label(lo, hi):
    return f'≥{lo}' if hi is None else f'{lo}–{hi}'


def analyse(directions, card):
    cuts = card['bins_for_figure_only']
    bands = [(lo, cuts[i + 1] if i + 1 < len(cuts) else None) for i, lo in enumerate(cuts)]
    perm = card['permutation']
    rows = []
    for d in directions:
        rows.append({'id': d['id'], 'domain': d['domain'], 'kind': d['kind'], 'gap': d['gap'],
                     'dG': d['dG'], 'dC': d['dC'], 'dG_ci95': d['dG_ci95'], 'dC_ci95': d['dC_ci95'], 'level': outcome_level(d['dG_ci95'], d['dC_ci95'])})
    out = {'directions': rows, 'kinds': {}}
    for kind in KINDS:
        rs = [r for r in rows if r['kind'] == kind]
        groups = defaultdict(list)
        for r in rs:
            groups[r['domain']].append((r['gap'], INDICATORS[card['indicator'][kind]](r)))
        test = stratified_spearman(dict(groups), perm['reps'], perm['seed'])
        exp = card['expected'][kind]['gap']
        sign_ok = test['point'] is not None and test['point'] != 0 and (test['point'] > 0) == (exp == 'positive')
        if test['p_two_sided'] is not None and test['p_two_sided'] < 0.05:
            verdict = '成立' if sign_ok else '相反'
        else:
            verdict = '看不出关系'
        by_band = []
        for lo, hi in bands:
            sel = [r for r in rs if r['gap'] >= lo and (hi is None or r['gap'] < hi)]
            c = Counter(r['level'] for r in sel)
            by_band.append({'band': band_label(lo, hi), 'n': len(sel), 'levels': {str(k): c.get(k, 0) for k in LEVELS},
                            'mean_dC': sum(r['dC'] for r in sel) / len(sel) if sel else None})
        by_domain = {}
        for dom in DOMAINS:
            c = Counter(r['level'] for r in rs if r['domain'] == dom)
            by_domain[dom] = {str(k): c.get(k, 0) for k in LEVELS}
        total = Counter(r['level'] for r in rs)
        out['kinds'][kind] = {'n': len(rs), 'indicator': card['indicator'][kind], 'test': test,
                              'indicator_count': sum(g[1] for gs in groups.values() for g in gs),
                              'expected': exp, 'verdict': verdict,
                              'levels': {str(k): total.get(k, 0) for k in LEVELS},
                              'by_gap_band': by_band, 'by_domain': by_domain}
    return out


def render(res, card, h08_path):
    lines = ['# H09：审阅结果的五档分布（事后、探索性）', '',
             card['statement_zh'], '',
             f'来源：`{h08_path}`（只读每个方向的 Δ_G、Δ_C 及 95% 区间）。档位：' +
             '；'.join(f'{k} {v}' for k, v in LEVEL_ZH.items()) + '。', '',
             '检验：' + card['test'], '']
    for name, a in res.items():
        lines += [f'## {name}', '']
        for kind, k in a['kinds'].items():
            t = k['test']
            pt = 'NA' if t['point'] is None else f"{t['point']:+.3f}"
            pv = 'NA' if t['p_two_sided'] is None else f"{t['p_two_sided']:.4f}"
            lines += [f"### {KINDS[kind]}（{k['n']} 个方向）", '',
                      f"指标：{INDICATOR_ZH[k['indicator']]}，共 {k['indicator_count']} 个方向。"
                      f"平均域内秩相关（差距 × 指标）{pt}，置换 p = {pv}"
                      f"（{t['reps']} 次）→ **{k['verdict']}**（预期 {k['expected']}）。", '',
                      '|差距（pp）|方向数|' + '|'.join(LEVEL_ZH[i] for i in LEVELS) + '|平均 Δ_C|',
                      '|---|---:|' + '---:|' * len(LEVELS) + '---:|']
            for b in k['by_gap_band']:
                mc = '' if b['mean_dC'] is None else f"{b['mean_dC']:+.1f}"
                lines.append(f"|{b['band']}|{b['n']}|" + '|'.join(str(b['levels'][str(i)]) for i in LEVELS) + f'|{mc}|')
            lines.append('|合计|' + str(k['n']) + '|' + '|'.join(str(k['levels'][str(i)]) for i in LEVELS) + '||')
            lines += ['', '|域|' + '|'.join(LEVEL_ZH[i] for i in LEVELS) + '|', '|---|' + '---:|' * len(LEVELS)]
            for dom, c in k['by_domain'].items():
                lines.append(f'|{dom}|' + '|'.join(str(c[str(i)]) for i in LEVELS) + '|')
            lines.append('')
    lines += ['分箱只用来列表和画图，检验用连续的差距。较弱者审较强者时只会出现 1、2、5 三档。',
              '第一版检验（差距 × 五档档位）及改动原因见卡片 first_version，如实保留。',
              '这是看过全部结果后提出的探索性假说，正文照此标注；登记好的 H08 照原样报告。', '']
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h08', required=True)
    ap.add_argument('--card', default=str(ROOT / 'claims/H09_outcome_levels.yaml'))
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    if (out_dir / 'H09_result.json').exists():
        sys.exit(f'{out_dir} already holds an H09 result; completed reports are never overwritten')
    card = yaml.safe_load(open(args.card, encoding='utf-8'))
    if card.get('analysis') != 'outcome_levels':
        sys.exit('card is not an outcome_levels card')
    h08 = json.load(open(args.h08, encoding='utf-8'))
    res = {name: analyse(a['directions'], card) for name, a in h08['analyses'].items()}
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {'claim': card['id'], 'analyses': res, 'h08_sha256': sha(args.h08),
               'claim_sha256': sha(args.card), 'script_sha256': sha(__file__),
               'level_names': {str(k): v for k, v in LEVEL_NAMES.items()}}
    (out_dir / 'H09_result.json').write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding='utf-8')
    (out_dir / 'H09_summary.md').write_text(render(res, card, args.h08), encoding='utf-8')
    print(out_dir / 'H09_summary.md')


if __name__ == '__main__':
    main()
