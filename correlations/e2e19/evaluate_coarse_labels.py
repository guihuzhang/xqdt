#!/usr/bin/env python3
"""
evaluate_coarse_labels.py
--------------------------
Evaluate XQDT outputs against E2E human coarse labels using the paper setting:
majority vote with lenient tie-breaking.

Three independent binary classification tasks (sample-level):
  is_ok        : model detects zero errors
                 vs majority vote says "ok"
  has_missing  : model detects MISSING or INCORRECT errors
                 vs majority vote says "missing" or "added missing"
  has_added    : model detects EXTRA errors
                 vs majority vote says "added" or "added missing"
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    from .worker_filter import load_excluded_workers
except ImportError:
    from worker_filter import load_excluded_workers


SCRIPT_DIR = Path(__file__).resolve().parent
CONVERTED = SCRIPT_DIR / 'human_ratings/converted.json'

PAPER_ORDER = [
    'gemma3_270m_20925',
    'gemma3_1b_9765',
    'gemma3_4b_6975',
    'gemma3_12b_4960',
    'qwen3_0.6b_11160',
    'qwen3_1.7b_15345',
    'qwen3_4b_8370',
    'qwen3_8b_8680',
    'qwen3_14b_4960',
    'llama3.2_1b_9765',
    'llama3.2_3b_6975',
    'llama3.1_8b_8680',
]

DISPLAY_NAMES = {
    'gemma3_270m_20925': 'G3-270M',
    'gemma3_1b_9765': 'G3-1B',
    'gemma3_4b_6975': 'G3-4B',
    'gemma3_12b_4960': 'G3-12B',
    'qwen3_0.6b_11160': 'Q3-0.6B',
    'qwen3_1.7b_15345': 'Q3-1.7B',
    'qwen3_4b_8370': 'Q3-4B',
    'qwen3_8b_8680': 'Q3-8B',
    'qwen3_14b_4960': 'Q3-14B',
    'llama3.2_1b_9765': 'L3.2-1B',
    'llama3.2_3b_6975': 'L3.2-3B',
    'llama3.1_8b_8680': 'L3.1-8B',
}

EXCLUDED_WORKERS = load_excluded_workers()

MISSING_LABELS = {'missing', 'added missing'}
ADDED_LABELS = {'added', 'added missing'}


def _norm(lbl):
    return re.sub(r'\s+', ' ', str(lbl or '').strip().lower())


def detect_default_results_dir() -> Path:
    preferred = SCRIPT_DIR / 'xqdt_results'
    fallback = SCRIPT_DIR / 'xqdt_results_repaired'
    if preferred.exists():
        return preferred
    return fallback


def model_sort_key(path: Path):
    model_name = path.stem.replace('_results', '')
    try:
        return (0, PAPER_ORDER.index(model_name))
    except ValueError:
        return (1, model_name)


def get_gt_lenient(annots):
    labels = []
    for a in annots:
        if str(a['worker_id']) in EXCLUDED_WORKERS:
            continue
        if a['fine_score'] > 100:
            continue
        lbl = _norm(a.get('coarse_label', ''))
        if lbl:
            labels.append(lbl)

    n = len(labels)
    if n == 0:
        return None, None, None, 0

    n_ok = sum(1 for l in labels if l == 'ok')
    n_miss = sum(1 for l in labels if l in MISSING_LABELS)
    n_add = sum(1 for l in labels if l in ADDED_LABELS)

    def _vote(n_pos, total, tie_to_pos):
        n_neg = total - n_pos
        if n_pos > n_neg:
            return True
        if n_pos < n_neg:
            return False
        return tie_to_pos

    gt_ok = _vote(n_ok, n, tie_to_pos=True)
    gt_miss = _vote(n_miss, n, tie_to_pos=False)
    gt_add = _vote(n_add, n, tie_to_pos=False)
    return gt_ok, gt_miss, gt_add, n


def get_pred(result):
    ec = result['error_counts']
    n_missing = ec.get('missing', 0)
    n_extra = ec.get('extra', 0)
    n_incorrect = ec.get('incorrect', 0)

    pred_ok = (n_missing == 0 and n_extra == 0 and n_incorrect == 0)
    pred_has_missing = (n_missing > 0 or n_incorrect > 0)
    pred_has_added = (n_extra > 0)
    return pred_ok, pred_has_missing, pred_has_added


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f


def evaluate(results, converted):
    expected = {(str(mr_id), system) for mr_id, record in converted.items() for system in record['systems']}
    actual = [(str(row['mr_id']), row['sys_name']) for row in results]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Prediction keys must exactly match the input MR/system pairs without duplicates")
    counts = {
        'ok': {'tp': 0, 'fp': 0, 'fn': 0},
        'miss': {'tp': 0, 'fp': 0, 'fn': 0},
        'add': {'tp': 0, 'fp': 0, 'fn': 0},
    }
    n_valid = 0
    n_skipped = 0
    gt_dist = {'ok': 0, 'miss': 0, 'add': 0}

    for result in results:
        mr_id = str(result['mr_id'])
        sys_name = result['sys_name']
        annots = (converted
                  .get(mr_id, {})
                  .get('systems', {})
                  .get(sys_name, {})
                  .get('quality', []))

        gt_ok, gt_miss, gt_add, n_votes = get_gt_lenient(annots)
        if n_votes == 0:
            n_skipped += 1
            continue

        pred_ok, pred_miss, pred_add = get_pred(result)
        n_valid += 1

        gt_dist['ok'] += int(gt_ok)
        gt_dist['miss'] += int(gt_miss)
        gt_dist['add'] += int(gt_add)

        for task, gt, pred in (
            ('ok', gt_ok, pred_ok),
            ('miss', gt_miss, pred_miss),
            ('add', gt_add, pred_add),
        ):
            if gt and pred:
                counts[task]['tp'] += 1
            elif (not gt) and pred:
                counts[task]['fp'] += 1
            elif gt and (not pred):
                counts[task]['fn'] += 1

    metrics = {}
    ps, rs, f1s = [], [], []
    for task in ('ok', 'miss', 'add'):
        c = counts[task]
        p, r, f = prf(c['tp'], c['fp'], c['fn'])
        metrics[task] = {'p': p, 'r': r, 'f1': f}
        ps.append(p)
        rs.append(r)
        f1s.append(f)

    metrics['macro_p'] = float(np.mean(ps))
    metrics['macro_r'] = float(np.mean(rs))
    metrics['macro_f1'] = float(np.mean(f1s))
    metrics['n_valid'] = n_valid
    metrics['n_skipped'] = n_skipped
    metrics['gt_dist'] = gt_dist
    return metrics


def fmt_prf(metrics, task):
    p, r, f = metrics[task]['p'], metrics[task]['r'], metrics[task]['f1']
    return f'{p:.3f} {r:.3f} {f:.3f}'


def fmt_macro(metrics):
    return f"{metrics['macro_p']:.3f} {metrics['macro_r']:.3f} {metrics['macro_f1']:.3f}"


def build_summary(models):
    payload = {
        'setting': 'majority_lenient',
        'paper_table_fields': {
            'is_ok': ['P', 'R', 'F1'],
            'has_missing': ['P', 'R', 'F1'],
            'has_added': ['P', 'R', 'F1'],
            'macro': ['P', 'R', 'F1'],
        },
        'models': {},
    }
    for model_name, metrics in models:
        payload['models'][model_name] = {
            'display_name': DISPLAY_NAMES.get(model_name, model_name),
            'n_valid': metrics['n_valid'],
            'n_skipped': metrics['n_skipped'],
            'gt_dist': metrics['gt_dist'],
            'is_ok': metrics['ok'],
            'has_missing': metrics['miss'],
            'has_added': metrics['add'],
            'macro': {
                'p': metrics['macro_p'],
                'r': metrics['macro_r'],
                'f1': metrics['macro_f1'],
            },
            'paper_row_percent': {
                'is_ok': {
                    'P': round(metrics['ok']['p'] * 100, 1),
                    'R': round(metrics['ok']['r'] * 100, 1),
                    'F1': round(metrics['ok']['f1'] * 100, 1),
                },
                'has_missing': {
                    'P': round(metrics['miss']['p'] * 100, 1),
                    'R': round(metrics['miss']['r'] * 100, 1),
                    'F1': round(metrics['miss']['f1'] * 100, 1),
                },
                'has_added': {
                    'P': round(metrics['add']['p'] * 100, 1),
                    'R': round(metrics['add']['r'] * 100, 1),
                    'F1': round(metrics['add']['f1'] * 100, 1),
                },
                'macro': {
                    'P': round(metrics['macro_p'] * 100, 1),
                    'R': round(metrics['macro_r'] * 100, 1),
                    'F1': round(metrics['macro_f1'] * 100, 1),
                },
            },
        }
    return payload


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate E2E coarse labels with the paper majority/lenient setting.'
    )
    parser.add_argument(
        '--results-dir',
        type=Path,
        default=detect_default_results_dir(),
        help='Directory containing *_results.json files.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Directory for the summary JSON and log. Defaults to results-dir.',
    )
    parser.add_argument(
        '--log-file',
        type=Path,
        default=None,
        help='Optional explicit log file path.',
    )
    return parser.parse_args()


def run():
    args = parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    output_dir = (args.output_dir.expanduser().resolve()
                  if args.output_dir is not None else results_dir)
    log_file = (args.log_file.expanduser().resolve()
                if args.log_file is not None
                else output_dir / f"coarse_label_eval_majority_lenient_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    summary_file = output_dir / 'table7_majority_lenient_summary.json'
    result_files = sorted(results_dir.glob('*_results.json'), key=model_sort_key)

    if not CONVERTED.exists():
        raise FileNotFoundError(f'Human ratings file not found: {CONVERTED}')
    if not results_dir.exists():
        raise FileNotFoundError(f'Results directory not found: {results_dir}')
    if not result_files:
        raise FileNotFoundError(f'No *_results.json files found under: {results_dir}')

    print('Loading converted.json ...')
    with open(CONVERTED, encoding='utf-8') as f:
        converted = json.load(f)

    lines = []

    def log(msg=''):
        print(msg)
        lines.append(msg)

    log('=' * 110)
    log('COARSE LABEL EVALUATION — majority / lenient')
    log('Tasks: is_ok | has_missing (MISSING+INCORRECT) | has_added (EXTRA)')
    log('Level: sample-level  |  Metrics: Precision / Recall / F1 per task + macro P/R/F1')
    log('Note: 41% of valid pairs are single-vote after filtering (majority vote = that vote)')
    log(f'Results dir: {results_dir}')
    log(f'Output dir:  {output_dir}')
    log('=' * 110)

    width = 17
    log('')
    h0 = (f"  {'Model':<28} |  n   |"
          f"  {'is_ok':^{width}}  |"
          f"  {'has_missing':^{width}}  |"
          f"  {'has_added':^{width}}  |"
          f"  {'macro':^{width}}  |")
    h1 = (f"  {'':<28} |      |"
          f"   P     R    F1   |"
          f"   P     R    F1   |"
          f"   P     R    F1   |"
          f"   P     R    F1   |")
    sep = '  ' + '-' * (len(h1) - 2)

    with open(result_files[0], encoding='utf-8') as f:
        ref_results = json.load(f)
    ref_metrics = evaluate(ref_results, converted)
    gt_dist = ref_metrics['gt_dist']
    n_valid = ref_metrics['n_valid']

    log('  [majority / lenient]')
    log(h0)
    log(h1)
    log(sep)
    log(f"  GT distribution ({n_valid} valid pairs, {ref_metrics['n_skipped']} skipped):"
        f"  ok={gt_dist['ok']}({100 * gt_dist['ok'] / n_valid:.0f}%)"
        f"  has_missing={gt_dist['miss']}({100 * gt_dist['miss'] / n_valid:.0f}%)"
        f"  has_added={gt_dist['add']}({100 * gt_dist['add'] / n_valid:.0f}%)")
    log(sep)

    models = []
    for result_file in result_files:
        model_name = result_file.stem.replace('_results', '')
        with open(result_file, encoding='utf-8') as f:
            results = json.load(f)
        metrics = evaluate(results, converted)
        models.append((model_name, metrics))
        display_name = DISPLAY_NAMES.get(model_name, model_name)
        row = (f"  {display_name:<28} | {metrics['n_valid']:>4} |"
               f"  {fmt_prf(metrics, 'ok')}  |"
               f"  {fmt_prf(metrics, 'miss')}  |"
               f"  {fmt_prf(metrics, 'add')}  |"
               f"  {fmt_macro(metrics)}  |")
        log(row)

    log('')
    log('=' * 110)
    log('majority/lenient : ok-tie→ok; missing/added-tie→no-error')
    log('Tie rates (after filtering): ok=7.6%, has_missing=6.9%, has_added=2.5%')

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(build_summary(models), f, indent=2, ensure_ascii=False)

    print(f'\nLog saved: {log_file}')
    print(f'Summary saved: {summary_file}')


if __name__ == '__main__':
    run()
