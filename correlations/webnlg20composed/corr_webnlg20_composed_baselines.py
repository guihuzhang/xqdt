#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

from tqdm import tqdm

from corr_webnlg20_composed import (
    DEFAULT_ANCHOR_EVAL,
    DEFAULT_EVAL_RESULTS_DIR,
    DEFAULT_HUMAN_JSON,
    DEFAULT_LOG_DIR,
    DEFAULT_METRIC_RESULTS_DIR,
    RANDOM_SEED,
    build_alignment,
    build_bootstrap_indices,
    build_composed_human,
    compute_flat_pooled,
    compute_system_level,
    compute_text_level2,
    format_summary,
    load_metric_matrix,
    relative_path,
)

DEFAULT_BOOTSTRAPPING = 1000
DEFAULT_OUTPUT_SUFFIX = 'baselines'

BASELINE_SCORE_MAP = {
    'monolr-precision': 'p',
    'monolr-recall': 'r',
    'monolr-f1': 'f1',
    'nli_based_p': 'p',
    'nli_based_r': 'r',
    'nli_based_f1': 'f1',
    'questeval_p': 'p',
    'questeval_r': 'r',
    'questeval': 'f1',
    'fact-spotter': 'r',
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bootstrapping', type=int, default=DEFAULT_BOOTSTRAPPING)
    parser.add_argument('--selected', nargs='*', default=None, help='Specific baseline folder names to run')
    parser.add_argument('--output-suffix', default=DEFAULT_OUTPUT_SUFFIX)
    parser.add_argument('--human-json', type=Path, default=DEFAULT_HUMAN_JSON)
    parser.add_argument('--eval-results-dir', type=Path, default=DEFAULT_EVAL_RESULTS_DIR)
    parser.add_argument('--metric-results-dir', type=Path, default=DEFAULT_METRIC_RESULTS_DIR)
    parser.add_argument('--log-dir', type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument('--anchor-eval', default=DEFAULT_ANCHOR_EVAL)
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    bootstrapping = args.bootstrapping
    selected_names = set(args.selected) if args.selected else None
    suffix = args.output_suffix.strip() or DEFAULT_OUTPUT_SUFFIX
    human_json = args.human_json
    eval_results_dir = args.eval_results_dir
    metric_results_dir = args.metric_results_dir
    log_dir = args.log_dir
    anchor_eval = args.anchor_eval.strip()
    if not anchor_eval:
        raise ValueError('Anchor eval file name must be non-empty')

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f'composed_corr_{suffix}.log'
    json_path = log_dir / f'composed_corr_results_{suffix}.json'

    aligned_items, _, human_to_metric_names = build_alignment(
        human_json=human_json,
        eval_results_dir=eval_results_dir,
        anchor_eval=anchor_eval,
    )
    systems = sorted(human_to_metric_names.values())
    entry_ids = [x['entry_id'] for x in aligned_items]
    composed_human = build_composed_human(aligned_items, human_to_metric_names)
    selected_bootstrap = build_bootstrap_indices(len(aligned_items), bootstrapping)

    folders = []
    for folder_name, score_key in BASELINE_SCORE_MAP.items():
        if selected_names is not None and folder_name not in selected_names:
            continue
        folder = metric_results_dir / folder_name
        if not folder.exists():
            raise FileNotFoundError(f'Baseline metric folder not found: {folder}')
        folders.append((folder_name, score_key, folder))

    results = {
        'meta': {
            'timestamp': str(datetime.datetime.now()),
            'n_graphs': len(aligned_items),
            'n_systems': len(systems),
            'n_points_flat': len(aligned_items) * len(systems),
            'bootstrapping': bootstrapping,
            'seed': RANDOM_SEED,
            'systems': systems,
            'entry_ids': entry_ids,
            'baseline_score_map': BASELINE_SCORE_MAP,
            'human_json': relative_path(human_json),
            'eval_results_dir': relative_path(eval_results_dir),
            'metric_results_dir': relative_path(metric_results_dir),
            'anchor_eval': anchor_eval,
        },
        'baselines': {},
    }

    with open(log_path, 'w', encoding='utf-8') as log_f:
        log_f.write('=' * 80 + '\n')
        log_f.write('WebNLG 2020 composed correlation - baselines\n')
        log_f.write('=' * 80 + '\n')
        log_f.write(f'Time: {datetime.datetime.now()}\n')
        log_f.write(f'Valid graphs: {len(aligned_items)}\n')
        log_f.write(f'Systems: {len(systems)}\n')
        log_f.write(f'Total pooled points: {len(aligned_items) * len(systems)}\n')
        log_f.write(f'Bootstrapping: {bootstrapping}\n')
        log_f.write('Mappings:\n')
        for folder_name, score_key in BASELINE_SCORE_MAP.items():
            log_f.write(f'  {folder_name} -> {score_key}\n')
        log_f.write('\n')

        for folder_name, score_key, folder in tqdm(folders, desc='baseline folders'):
            model_scores = load_metric_matrix(folder, entry_ids)
            missing_systems = sorted(set(systems) - set(model_scores.keys()))
            if missing_systems:
                raise ValueError(f'{folder_name} missing systems: {missing_systems}')
            model_scores = {system: model_scores[system] for system in systems}
            human_scores = {system: composed_human[system][score_key] for system in systems}

            print(f"\n[{folder_name}]")
            print(f"metric: {folder_name}")
            print(f"score: {score_key}")
            print(f"folder: {folder_name}")
            system_level = compute_system_level(model_scores, human_scores, selected_bootstrap)
            text_level2 = compute_text_level2(model_scores, human_scores, selected_bootstrap)
            flat_pooled = compute_flat_pooled(model_scores, human_scores)

            result = {
                'score_key': score_key,
                'metric_folder': relative_path(folder),
                'system_level': system_level,
                'text_level2': text_level2,
                'flat_pooled': flat_pooled,
            }
            results['baselines'][folder_name] = result

            log_f.write(f'[{folder_name}]\n')
            log_f.write(f'metric: {folder_name}\n')
            log_f.write(f'score: {score_key}\n')
            log_f.write(f'folder: {folder_name}\n')
            for section_name, summary in (
                ('System level', system_level),
                ('Text level 2', text_level2),
                ('Flat pooled', flat_pooled),
            ):
                lines = format_summary(section_name, summary)
                for line in lines:
                    print(line)
                    log_f.write(line + '\n')
            log_f.write('\n')
            log_f.flush()

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f'\nSaved log: {log_path}')
    print(f'Saved json: {json_path}')


if __name__ == '__main__':
    main()
