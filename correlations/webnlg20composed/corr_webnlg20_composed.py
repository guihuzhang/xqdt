#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

from scipy.stats import kendalltau, pearsonr, spearmanr
from tqdm import tqdm

RANDOM_SEED = 10
DEFAULT_BOOTSTRAPPING = 1000
MISSING_ENTRY_ID = 106

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_HUMAN_JSON = BASE_DIR / 'human-annotations' / 'human_eval_webnlg_20.json'
DEFAULT_EVAL_RESULTS_DIR = BASE_DIR / 'xqdt_results'
DEFAULT_METRIC_RESULTS_DIR = BASE_DIR / 'metric_results'
DEFAULT_LOG_DIR = BASE_DIR / 'correlation_logs' / 'webnlg2020'
DEFAULT_ANCHOR_EVAL = 'qwen3_8b_7479.json'
DEFAULT_LOG_NAME = 'composed_corr.log'
DEFAULT_JSON_NAME = 'composed_corr_results.json'

SYSTEM_NAME_MAP = {
    'Amazon_AI_(Shanghai)': 'Amazon_AI_Shanghai',
}


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f'Required input not found: {path}')
    return path


def relative_path(path: Path) -> str:
    return os.path.relpath(path, BASE_DIR)


def load_json(path: Path):
    with open(require_file(path), 'r', encoding='utf-8') as f:
        return json.load(f)


def harmonic_mean(a: float, b: float) -> float:
    if a + b == 0:
        return 0.0
    return (2 * a * b) / (a + b)


def compute_rmse(xs: List[float], ys: List[float]) -> float:
    if len(xs) != len(ys):
        raise ValueError('RMSE inputs must have the same length')
    if not xs:
        return 0.0
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(xs, ys)) / len(xs))


def build_bootstrap_indices(n_samples: int, n_bootstrap: int) -> List[List[int]]:
    random.seed(RANDOM_SEED)
    indices = list(range(n_samples))
    return [random.choices(indices, k=n_samples) for _ in range(n_bootstrap)]


def correlation_triplet(xs: List[float], ys: List[float]) -> Dict[str, Dict[str, float] | None]:
    def safe_corr(fn, xvals, yvals):
        try:
            result = fn(xvals, yvals)
        except Exception:
            return None
        if hasattr(result, 'statistic'):
            stat = result.statistic
            pvalue = result.pvalue
        elif hasattr(result, 'correlation'):
            stat = result.correlation
            pvalue = result.pvalue
        else:
            stat = result[0]
            pvalue = result[1]
        if stat is None or (isinstance(stat, float) and math.isnan(stat)):
            return None
        if pvalue is None or (isinstance(pvalue, float) and math.isnan(pvalue)):
            return None
        return {'stat': float(stat), 'pvalue': float(pvalue)}

    return {
        'pearson': safe_corr(pearsonr, xs, ys),
        'spearman': safe_corr(spearmanr, xs, ys),
        'kendall': safe_corr(kendalltau, xs, ys),
    }


def summarize_boot(values: List[float]) -> Dict[str, float] | None:
    if not values:
        return None
    values = sorted(values)
    return {
        'mean': sum(values) / len(values),
        'samples': len(values),
        'lo': values[math.floor(len(values) * 0.05)],
        'hi': values[math.floor(len(values) * 0.95)],
    }


def build_alignment(
    human_json: Path,
    eval_results_dir: Path,
    anchor_eval: str,
) -> Tuple[List[dict], Dict[str, List[int]], Dict[str, str]]:
    human_items = load_json(human_json)
    eval_path = require_file(eval_results_dir / anchor_eval)
    eval_items = load_json(eval_path)

    aligned_items: List[dict] = []
    system_to_ids: Dict[str, List[int]] = {}
    human_to_metric_names: Dict[str, str] = {}

    pred_keys = [k for k in human_items[0].keys() if k.endswith('-pred')]
    for pred_key in pred_keys:
        human_system = pred_key[:-5]
        metric_system = SYSTEM_NAME_MAP.get(human_system, human_system)
        system_to_ids[metric_system] = []
        human_to_metric_names[human_system] = metric_system

    anchor_human = 'Amazon_AI_(Shanghai)'
    anchor_metric = human_to_metric_names[anchor_human]
    anchor_eval = [
        item for item in eval_items
        if item['team'] == anchor_metric and item['entry_id'] != MISSING_ENTRY_ID
    ]

    anchor_index: Dict[str, List[int]] = {}
    for item in anchor_eval:
        anchor_index.setdefault(item['text'].strip(), []).append(item['entry_id'])

    eval_by_team_and_id: Dict[Tuple[str, int], dict] = {}
    for item in eval_items:
        if item['entry_id'] == MISSING_ENTRY_ID:
            continue
        eval_by_team_and_id[(item['team'], item['entry_id'])] = item

    for obj in human_items:
        anchor_text = obj[f'{anchor_human}-pred'].strip()
        candidate_ids = anchor_index.get(anchor_text, [])
        if len(candidate_ids) != 1:
            raise ValueError(
                f'Anchor alignment failed for text: {anchor_text!r}; candidates={candidate_ids}'
            )
        chosen_entry_id = candidate_ids[0]

        for pred_key in pred_keys:
            human_system = pred_key[:-5]
            metric_system = human_to_metric_names[human_system]
            eval_item = eval_by_team_and_id.get((metric_system, chosen_entry_id))
            if eval_item is None:
                raise ValueError(
                    f'Missing eval item for system={metric_system}, entry_id={chosen_entry_id}'
                )
            expected_text = obj[pred_key].strip()
            actual_text = eval_item['text'].strip()
            if expected_text != actual_text:
                raise ValueError(
                    f'Text mismatch for system={metric_system}, entry_id={chosen_entry_id}'
                )

        obj = dict(obj)
        obj['entry_id'] = chosen_entry_id
        aligned_items.append(obj)

        for pred_key in pred_keys:
            human_system = pred_key[:-5]
            metric_system = human_to_metric_names[human_system]
            system_to_ids[metric_system].append(chosen_entry_id)

    aligned_items.sort(key=lambda x: x['entry_id'])
    for metric_system, entry_ids in system_to_ids.items():
        if len(entry_ids) != len(aligned_items):
            raise ValueError(f'{metric_system} has {len(entry_ids)} aligned ids, expected {len(aligned_items)}')
        if sorted(entry_ids) != [x['entry_id'] for x in aligned_items]:
            raise ValueError(f'Entry alignment mismatch for system: {metric_system}')

    return aligned_items, system_to_ids, human_to_metric_names


def build_composed_human(aligned_items: List[dict], human_to_metric_names: Dict[str, str]):
    composed: Dict[str, Dict[str, List[float]]] = {}
    for human_system, metric_system in human_to_metric_names.items():
        p_vals: List[float] = []
        r_vals: List[float] = []
        f1_vals: List[float] = []
        for obj in aligned_items:
            c = float(obj[f'{human_system}-correctness']) / 100.0
            rel = float(obj[f'{human_system}-relevance']) / 100.0
            d = float(obj[f'{human_system}-datacoverage']) / 100.0
            p = c * rel
            r = d
            f1 = harmonic_mean(p, r)
            p_vals.append(p)
            r_vals.append(r)
            f1_vals.append(f1)
        composed[metric_system] = {'p': p_vals, 'r': r_vals, 'f1': f1_vals}
    return composed


def load_metric_matrix(metric_folder: Path, entry_ids: List[int]) -> Dict[str, List[float]]:
    matrix: Dict[str, List[float]] = {}
    for txt_path in sorted(metric_folder.glob('*.txt')):
        system_name = txt_path.stem
        with open(txt_path, 'r', encoding='utf-8') as f:
            values = [float(line.strip()) for line in f if line.strip()]
        if len(values) <= max(entry_ids):
            raise ValueError(
                f'{txt_path} has {len(values)} scores but needs index {max(entry_ids)}'
            )
        matrix[system_name] = [values[i] for i in entry_ids]
    return matrix


def compute_system_level(
    model_scores: Dict[str, List[float]],
    human_scores: Dict[str, List[float]],
    selected_bootstrap: List[List[int]],
) -> Dict[str, Dict[str, float] | None]:
    pear_boot: List[float] = []
    sp_boot: List[float] = []
    k_boot: List[float] = []
    rmse_boot: List[float] = []
    systems = sorted(model_scores.keys())

    for selected in tqdm(selected_bootstrap, desc='system bootstrap', leave=False):
        x_means: List[float] = []
        y_means: List[float] = []
        for system in systems:
            x_means.append(sum(model_scores[system][i] for i in selected) / len(selected))
            y_means.append(sum(human_scores[system][i] for i in selected) / len(selected))
        corr = correlation_triplet(x_means, y_means)
        rmse_boot.append(compute_rmse(x_means, y_means))
        if corr['pearson'] and corr['pearson']['pvalue'] < 0.05:
            pear_boot.append(corr['pearson']['stat'])
        if corr['spearman'] and corr['spearman']['pvalue'] < 0.05:
            sp_boot.append(corr['spearman']['stat'])
        if corr['kendall'] and corr['kendall']['pvalue'] < 0.05:
            k_boot.append(corr['kendall']['stat'])

    return {
        'pearson': summarize_boot(pear_boot),
        'spearman': summarize_boot(sp_boot),
        'kendall': summarize_boot(k_boot),
        'rmse': summarize_boot(rmse_boot),
    }


def compute_text_level2(
    model_scores: Dict[str, List[float]],
    human_scores: Dict[str, List[float]],
    selected_bootstrap: List[List[int]],
) -> Dict[str, Dict[str, float] | None]:
    pear_boot: List[float] = []
    sp_boot: List[float] = []
    k_boot: List[float] = []
    rmse_boot: List[float] = []
    systems = sorted(model_scores.keys())
    size_test = len(next(iter(model_scores.values())))
    correlations_by_graph = []
    errors_by_graph = []
    for position in range(size_test):
        xs = [model_scores[system][position] for system in systems]
        ys = [human_scores[system][position] for system in systems]
        correlations_by_graph.append(correlation_triplet(xs, ys))
        errors_by_graph.append(compute_rmse(xs, ys))

    for selected in tqdm(selected_bootstrap, desc='text2 bootstrap', leave=False):
        pear_iter: List[float] = []
        sp_iter: List[float] = []
        k_iter: List[float] = []
        rmse_iter: List[float] = []
        for position in selected:
            corr = correlations_by_graph[position]
            rmse_iter.append(errors_by_graph[position])
            if corr['pearson'] and corr['pearson']['pvalue'] < 0.05:
                pear_iter.append(corr['pearson']['stat'])
            if corr['spearman'] and corr['spearman']['pvalue'] < 0.05:
                sp_iter.append(corr['spearman']['stat'])
            if corr['kendall'] and corr['kendall']['pvalue'] < 0.05:
                k_iter.append(corr['kendall']['stat'])
        if pear_iter:
            pear_boot.append(sum(pear_iter) / len(pear_iter))
        if sp_iter:
            sp_boot.append(sum(sp_iter) / len(sp_iter))
        if k_iter:
            k_boot.append(sum(k_iter) / len(k_iter))
        if rmse_iter:
            rmse_boot.append(sum(rmse_iter) / len(rmse_iter))

    return {
        'pearson': summarize_boot(pear_boot),
        'spearman': summarize_boot(sp_boot),
        'kendall': summarize_boot(k_boot),
        'rmse': summarize_boot(rmse_boot),
    }


def compute_flat_pooled(
    model_scores: Dict[str, List[float]],
    human_scores: Dict[str, List[float]],
) -> Dict[str, Dict[str, float] | None]:
    systems = sorted(model_scores.keys())
    xs: List[float] = []
    ys: List[float] = []
    for system in systems:
        xs.extend(model_scores[system])
        ys.extend(human_scores[system])
    corr = correlation_triplet(xs, ys)
    return {
        'pearson': corr['pearson'],
        'spearman': corr['spearman'],
        'kendall': corr['kendall'],
        'rmse': {'stat': compute_rmse(xs, ys)},
        'n_points': len(xs),
    }


def describe_dt_folder(folder_name: str) -> Dict[str, str]:
    parts = folder_name.split('_')
    if len(parts) >= 4 and parts[0] == 'dt':
        score = parts[-1]
        model = '_'.join(parts[1:-1])
        return {'folder': folder_name, 'model': model, 'score': score}
    return {'folder': folder_name, 'model': folder_name, 'score': 'unknown'}


def format_summary(name: str, summary: Dict[str, Dict[str, float] | None]) -> List[str]:
    lines = [name]
    for metric_key in ('pearson', 'spearman', 'kendall', 'rmse'):
        item = summary.get(metric_key)
        if item is None:
            lines.append(f'  {metric_key}: nan')
        elif 'mean' in item:
            lines.append(
                f"  {metric_key}: {item['mean']:.4f} "
                f"(samples={item['samples']}, ci=[{item['lo']:.4f}, {item['hi']:.4f}])"
            )
        elif metric_key == 'rmse':
            lines.append(f"  rmse: {item['stat']:.4f}")
        else:
            lines.append(
                f"  {metric_key}: {item['stat']:.4f} (p={item['pvalue']:.4g})"
            )
    return lines


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bootstrapping', type=int, default=DEFAULT_BOOTSTRAPPING)
    parser.add_argument('--selected', nargs='*', default=None, help='Specific dt_* metric folder names to run')
    parser.add_argument('--output-suffix', default='', help='Suffix added to output log/json file names')
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
    output_suffix = args.output_suffix.strip()
    human_json = require_file(args.human_json)
    eval_results_dir = require_file(args.eval_results_dir)
    metric_results_dir = require_file(args.metric_results_dir)
    log_dir = args.log_dir
    anchor_eval = args.anchor_eval.strip()
    if not anchor_eval:
        raise ValueError('Anchor eval file name must be non-empty')

    log_dir.mkdir(parents=True, exist_ok=True)
    suffix = f'_{output_suffix}' if output_suffix else ''
    log_path = log_dir / f'composed_corr{suffix}.log'
    json_path = log_dir / f'composed_corr_results{suffix}.json'

    print('=' * 80)
    print('WebNLG 2020 composed correlation')
    print('=' * 80)
    print(f'Human JSON: {human_json}')
    print(f'Evaluation results: {eval_results_dir}')
    print(f'Metric results: {metric_results_dir}')
    print(f'Anchor eval: {anchor_eval}')
    print(f'Missing entry ignored: {MISSING_ENTRY_ID}')

    aligned_items, system_to_ids, human_to_metric_names = build_alignment(
        human_json=human_json,
        eval_results_dir=eval_results_dir,
        anchor_eval=anchor_eval,
    )
    systems = sorted(human_to_metric_names.values())
    entry_ids = [x['entry_id'] for x in aligned_items]
    composed_human = build_composed_human(aligned_items, human_to_metric_names)
    selected_bootstrap = build_bootstrap_indices(len(aligned_items), bootstrapping)

    print(f'Valid graphs: {len(aligned_items)}')
    print(f'Systems: {len(systems)}')
    print(f'Total pooled points: {len(aligned_items) * len(systems)}')

    dt_folders = sorted(
        folder for folder in metric_results_dir.iterdir()
        if folder.is_dir() and folder.name.startswith('dt_')
    )
    if selected_names is not None:
        dt_folders = [folder for folder in dt_folders if folder.name in selected_names]

    all_results = {
        'meta': {
            'timestamp': str(datetime.datetime.now()),
            'n_graphs': len(aligned_items),
            'n_systems': len(systems),
            'n_points_flat': len(aligned_items) * len(systems),
            'ignored_entry_id': MISSING_ENTRY_ID,
            'systems': systems,
            'entry_ids': entry_ids,
            'bootstrapping': bootstrapping,
            'seed': RANDOM_SEED,
            'human_json': relative_path(human_json),
            'eval_results_dir': relative_path(eval_results_dir),
            'metric_results_dir': relative_path(metric_results_dir),
            'anchor_eval': anchor_eval,
        },
        'checkpoints': {},
    }

    with open(log_path, 'w', encoding='utf-8') as log_f:
        log_f.write('=' * 80 + '\n')
        log_f.write('WebNLG 2020 composed correlation\n')
        log_f.write('=' * 80 + '\n')
        log_f.write(f'Time: {datetime.datetime.now()}\n')
        log_f.write(f'Valid graphs: {len(aligned_items)}\n')
        log_f.write(f'Systems: {len(systems)}\n')
        log_f.write(f'Total pooled points: {len(aligned_items) * len(systems)}\n')
        log_f.write(f'Ignored entry_id: {MISSING_ENTRY_ID}\n')
        log_f.write(f'Bootstrapping: {bootstrapping}\n')
        if selected_names is not None:
            log_f.write(f'Selected folders: {sorted(selected_names)}\n')
        log_f.write('\n')

        for folder in tqdm(dt_folders, desc='metric folders'):
            name = folder.name
            if name.endswith('_p'):
                score_key = 'p'
            elif name.endswith('_r'):
                score_key = 'r'
            elif name.endswith('_f1'):
                score_key = 'f1'
            else:
                continue

            model_scores = load_metric_matrix(folder, entry_ids)
            missing_systems = sorted(set(systems) - set(model_scores.keys()))
            if missing_systems:
                raise ValueError(f'{name} missing systems: {missing_systems}')
            model_scores = {system: model_scores[system] for system in systems}
            human_scores = {system: composed_human[system][score_key] for system in systems}

            meta = describe_dt_folder(name)
            print(f"\n[{name}]")
            print(f"model: {meta['model']}")
            print(f"score: {meta['score']}")
            print(f"folder: {meta['folder']}")
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
            all_results['checkpoints'][name] = result

            log_f.write(f'[{name}]\n')
            log_f.write(f"model: {meta['model']}\n")
            log_f.write(f"score: {meta['score']}\n")
            log_f.write(f"folder: {meta['folder']}\n")
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
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f'\nSaved log: {log_path}')
    print(f'Saved json: {json_path}')


if __name__ == '__main__':
    main()
