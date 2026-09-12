#!/usr/bin/env python3
"""
table19_e2e_in_vs_cross_domain.py
----------------------------------
Reproduce the paper's Table 19 (E2E fine-grained human-judgement correlations)
for the E2E-trained (in-domain) verifiers, and compute the analogous numbers for
the WebNLG-only (cross-domain) verifiers, using the SAME
drop_gt100_all+A+C+D+E+F worker-filtering strategy (see e2e_correlation_light.py
for details and provenance).

Without this filtering step, correlations computed directly on the raw
*_results.json files do NOT match the paper (e.g. Qwen3-8B in-domain system-level
Pearson r comes out ~15% instead of the published 79.8%).

Outputs:
  - table19_in_vs_cross_domain_summary.json  (all 12 models, in + cross domain)
  - table19_in_vs_cross_domain_qwen.md       (Qwen-only markdown table,
                                              one in-domain + one cross-domain
                                              row per model)
"""

import argparse
import json
from pathlib import Path

from e2e_correlation_light import compute_correlations_for_metric, rebuild_human_quality_scores

BASE = Path(__file__).resolve().parents[2]
CONVERTED = Path(__file__).resolve().parent / 'human_ratings/converted.json'
INDOMAIN_DIR = Path(__file__).resolve().parent / 'xqdt_results_repaired'
CROSS_DIR = (
    BASE / 'verifier_train_eval_e2e19/cross_dataset_results/webnlg_to_e2e_human'
)
OUT_JSON = Path(__file__).resolve().parent / 'table19_in_vs_cross_domain_summary.json'
OUT_MD = Path(__file__).resolve().parent / 'table19_in_vs_cross_domain_qwen.md'

# (display name, cross-domain file, in-domain file)
MODEL_PAIRS = [
    ('Gemma3-270M', 'gemma3_270m_16620_results.json', 'gemma3_270m_20925_results.json'),
    ('Gemma3-1B',   'gemma3_1b_6648_results.json',    'gemma3_1b_9765_results.json'),
    ('Gemma3-4B',   'gemma3_4b_6648_results.json',    'gemma3_4b_6975_results.json'),
    ('Gemma3-12B',  'gemma3_12b_4432_results.json',   'gemma3_12b_4960_results.json'),
    ('Qwen3-0.6B',  'qwen3_0.6b_8864_results.json',   'qwen3_0.6b_11160_results.json'),
    ('Qwen3-1.7B',  'qwen3_1.7b_6648_results.json',   'qwen3_1.7b_15345_results.json'),
    ('Qwen3-4B',    'qwen3_4b_4432_results.json',     'qwen3_4b_8370_results.json'),
    ('Qwen3-8B',    'qwen3_8b_7479_results.json',     'qwen3_8b_8680_results.json'),
    ('Qwen3-14B',   'qwen3_14b_6648_results.json',    'qwen3_14b_4960_results.json'),
    ('Llama3.2-1B', 'llama3_2_1b_8864_results.json',  'llama3.2_1b_9765_results.json'),
    ('Llama3.2-3B', 'llama3_2_3b_5540_results.json',  'llama3.2_3b_6975_results.json'),
    ('Llama3.1-8B', 'llama3_1_8b_5263_results.json',  'llama3.1_8b_8680_results.json'),
]

QWEN_MODELS = {'Qwen3-0.6B', 'Qwen3-1.7B', 'Qwen3-4B', 'Qwen3-8B', 'Qwen3-14B'}


def score(path, converted):
    results = json.load(open(path, encoding='utf-8'))
    rebuilt = rebuild_human_quality_scores(results, converted)
    return compute_correlations_for_metric(rebuilt, 'f1')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_json = args.output_dir / OUT_JSON.name
    output_md = args.output_dir / OUT_MD.name
    converted = json.load(open(CONVERTED, encoding='utf-8'))

    summary = {}
    for name, cross_file, in_file in MODEL_PAIRS:
        cross = score(CROSS_DIR / cross_file, converted)
        indomain = score(INDOMAIN_DIR / in_file, converted)
        summary[name] = {'cross_domain': cross, 'in_domain': indomain}

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f'Summary saved: {output_json}')

    def f1_row(entry):
        sl, tl = entry['system_level'], entry['text_level']
        return (
            sl['pearson'] * 100, sl['spearman'] * 100, sl['kendall'] * 100,
            tl['pearson'] * 100, tl['spearman'] * 100, tl['kendall'] * 100,
        )

    lines = [
        '| Model | Setting | System r | System ρ | System τ | Text r | Text ρ | Text τ |',
        '|---|---|---:|---:|---:|---:|---:|---:|',
    ]
    for name, _, _ in MODEL_PAIRS:
        if name not in QWEN_MODELS:
            continue
        sr, srho, stau, tr, trho, ttau = f1_row(summary[name]['in_domain'])
        lines.append(
            f'| {name} | In-domain | {sr:.1f} | {srho:.1f} | {stau:.1f} '
            f'| {tr:.1f} | {trho:.1f} | {ttau:.1f} |'
        )
        sr, srho, stau, tr, trho, ttau = f1_row(summary[name]['cross_domain'])
        lines.append(
            f'| {name} | Cross-domain | {sr:.1f} | {srho:.1f} | {stau:.1f} '
            f'| {tr:.1f} | {trho:.1f} | {ttau:.1f} |'
        )

    with open(output_md, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'Markdown saved: {output_md}')


if __name__ == '__main__':
    main()
