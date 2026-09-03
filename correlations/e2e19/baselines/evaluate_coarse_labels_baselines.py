#!/usr/bin/env python3
"""
evaluate_coarse_labels_baselines.py
-------------------------------------
Evaluate NLI / QuestEval / FactSpotter baselines on E2E coarse labels using
the paper setting: majority vote with lenient tie-breaking.
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
E2E19_ROOT = SCRIPT_DIR.parent
CONVERTED = E2E19_ROOT / 'human_ratings/converted.json'
SCORES_FILE = SCRIPT_DIR / 'baseline_scores_e2e19.json'
sys.path.insert(0, str(E2E19_ROOT))

from worker_filter import load_excluded_workers

NLI_THRESHOLD = 0.5
QE_THRESHOLD = 0.5
FS_THRESHOLD = 0.5

DISPLAY_NAMES = {
    'NLI': 'NLI',
    'QuestEval': 'DQE',
    'FactSpotter': 'FS',
}

EXCLUDED_WORKERS = load_excluded_workers()

MISSING_LABELS = {'missing', 'added missing'}
ADDED_LABELS = {'added', 'added missing'}


def _norm(lbl):
    return re.sub(r'\s+', ' ', str(lbl or '').strip().lower())


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

    return (
        _vote(n_ok, n, True),
        _vote(n_miss, n, False),
        _vote(n_add, n, False),
        n,
    )


def pred_nli(scores):
    prec = scores.get('nli_prec')
    rec_list = scores.get('nli_rec_list')
    if prec is None:
        return None
    if rec_list:
        pred_miss = any(p < NLI_THRESHOLD for p in rec_list)
    else:
        rec = scores.get('nli_rec')
        if rec is None:
            return None
        pred_miss = rec < NLI_THRESHOLD

    pred_added = prec < NLI_THRESHOLD
    pred_ok = not pred_miss and not pred_added
    return pred_ok, pred_miss, pred_added


def pred_questeval(scores):
    prec = scores.get('questeval_prec')
    rec = scores.get('questeval_rec')
    if prec is None or rec is None:
        return None
    pred_miss = rec < QE_THRESHOLD
    pred_added = prec < QE_THRESHOLD
    pred_ok = not pred_miss and not pred_added
    return pred_ok, pred_miss, pred_added


def pred_factspotter(scores):
    fs_list = scores.get('factspotter_list')
    if fs_list:
        pred_miss = any(p < FS_THRESHOLD for p in fs_list)
    else:
        fs = scores.get('factspotter_electra')
        if fs is None:
            return None
        pred_miss = fs < FS_THRESHOLD

    pred_added = False
    pred_ok = not pred_miss
    return pred_ok, pred_miss, pred_added


BASELINES = [
    ('NLI', pred_nli, ''),
    ('QuestEval', pred_questeval, ' [mean]'),
    ('FactSpotter', pred_factspotter, ' [has_add=N/A]'),
]


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f


def evaluate(pairs, converted, pred_fn):
    counts = {k: {'tp': 0, 'fp': 0, 'fn': 0} for k in ('ok', 'miss', 'add')}
    n_valid = 0
    n_skipped = 0
    gt_dist = {'ok': 0, 'miss': 0, 'add': 0}

    for mr_id, sys_name, scores in pairs:
        annots = (converted
                  .get(str(mr_id), {})
                  .get('systems', {})
                  .get(sys_name, {})
                  .get('quality', []))

        gt_ok, gt_miss, gt_add, n_votes = get_gt_lenient(annots)
        if n_votes == 0:
            n_skipped += 1
            continue

        preds = pred_fn(scores)
        if preds is None:
            n_skipped += 1
            continue

        pred_ok, pred_miss, pred_add = preds
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
    p, r, f = metrics[task]['p'] * 100, metrics[task]['r'] * 100, metrics[task]['f1'] * 100
    return f'{p:5.1f} {r:5.1f} {f:5.1f}'


def fmt_macro(metrics):
    p, r, f = metrics['macro_p'] * 100, metrics['macro_r'] * 100, metrics['macro_f1'] * 100
    return f'{p:5.1f} {r:5.1f} {f:5.1f}'


def build_summary(metrics_by_name):
    payload = {
        'setting': 'majority_lenient',
        'models': {},
    }
    for name, metrics in metrics_by_name.items():
        payload['models'][name] = {
            'display_name': DISPLAY_NAMES.get(name, name),
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
        description='Evaluate E2E coarse labels for baselines with the paper majority/lenient setting.'
    )
    parser.add_argument(
        '--scores-file',
        type=Path,
        default=SCORES_FILE,
        help='Path to baseline_scores_e2e19.json.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Directory for the log and summary JSON. Defaults to the scores file directory.',
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
    scores_file = args.scores_file.expanduser().resolve()
    output_dir = (args.output_dir.expanduser().resolve()
                  if args.output_dir is not None else scores_file.parent)
    log_file = (args.log_file.expanduser().resolve()
                if args.log_file is not None
                else output_dir / f"coarse_label_eval_baselines_majority_lenient_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    summary_file = output_dir / 'table7_baselines_majority_lenient_summary.json'

    if not CONVERTED.exists():
        raise FileNotFoundError(f'Human ratings file not found: {CONVERTED}')
    if not scores_file.exists():
        raise FileNotFoundError(f'Baseline cache not found: {scores_file}')

    print('Loading converted.json ...')
    with open(CONVERTED, encoding='utf-8') as f:
        converted = json.load(f)

    print(f'Loading {scores_file.name} ...')
    with open(scores_file, encoding='utf-8') as f:
        scores_cache = json.load(f)

    n_nli_pertri = sum(1 for v in scores_cache.values() if 'nli_rec_list' in v)
    n_fs_pertri = sum(1 for v in scores_cache.values() if 'factspotter_list' in v)
    print(f'  Per-triple NLI:         {n_nli_pertri}/{len(scores_cache)}')
    print(f'  Per-triple FactSpotter: {n_fs_pertri}/{len(scores_cache)}')
    if n_nli_pertri < len(scores_cache) or n_fs_pertri < len(scores_cache):
        print('  ⚠️  Some per-triple lists are missing. Falling back to mean-based thresholding where needed.\n')

    pairs = []
    for key, scores in scores_cache.items():
        mr_id, sys_name = key.split('||', 1)
        pairs.append((mr_id, sys_name, scores))
    print(f'  {len(pairs)} pairs loaded\n')

    lines = []

    def log(msg=''):
        print(msg)
        lines.append(msg)

    cols = 116
    log('=' * cols)
    log('COARSE LABEL EVALUATION — BASELINES (majority / lenient)')
    log(f'  NLI:         per-triple any-fails, threshold={NLI_THRESHOLD} '
        f'(nli_rec_list coverage: {n_nli_pertri}/{len(scores_cache)})')
    log(f'  QuestEval:   mean-based, threshold={QE_THRESHOLD} '
        f'(questions not triple-aligned)')
    log(f'  FactSpotter: per-triple any-fails, threshold={FS_THRESHOLD} '
        f'(factspotter_list coverage: {n_fs_pertri}/{len(scores_cache)}); '
        f'has_added always False')
    log('  Worker filtering: drop_gt100_all + A+C+D+E+F')
    log(f'  Scores file: {scores_file}')
    log(f'  Output dir:  {output_dir}')
    log('=' * cols)

    h0 = (f"  {'Model':<22} |  n   |"
          f"    {'is_ok':^17}  |"
          f"    {'has_missing':^17}  |"
          f"    {'has_added':^17}  |"
          f"    {'macro':^17}  |")
    h1 = (f"  {'':<22} |      |"
          f"      P     R    F1    |"
          f"      P     R    F1    |"
          f"      P     R    F1    |"
          f"      P     R    F1    |")
    sep = '  ' + '-' * (len(h1) - 2)

    log('\n  [majority / lenient]')
    log(h0)
    log(h1)
    log(sep)

    ref_metrics = evaluate(pairs, converted, pred_nli)
    gt_dist = ref_metrics['gt_dist']
    n_valid = max(ref_metrics['n_valid'], 1)
    log(f"  GT ({ref_metrics['n_valid']} valid, {ref_metrics['n_skipped']} skipped):"
        f"  ok={gt_dist['ok']} ({100 * gt_dist['ok'] / n_valid:.0f}%)"
        f"  has_missing={gt_dist['miss']} ({100 * gt_dist['miss'] / n_valid:.0f}%)"
        f"  has_added={gt_dist['add']} ({100 * gt_dist['add'] / n_valid:.0f}%)")
    log(sep)

    summary_metrics = {}
    for name, pred_fn, note in BASELINES:
        metrics = evaluate(pairs, converted, pred_fn)
        summary_metrics[name] = metrics
        label = DISPLAY_NAMES.get(name, name) + note
        row = (f"  {label:<22} | {metrics['n_valid']:>4} |"
               f"    {fmt_prf(metrics, 'ok')}    |"
               f"    {fmt_prf(metrics, 'miss')}    |"
               f"    {fmt_prf(metrics, 'add')}    |"
               f"    {fmt_macro(metrics)}    |")
        log(row)

    log('')
    log('=' * cols)
    log('Notes:')
    log('  NLI has_missing: any(triple entailment prob < 0.5) across all source triples.')
    log('  NLI has_added:   mean nli_prec < 0.5 (single-direction, no per-triple decomposition).')
    log('  QuestEval: mean answerability; questions are generated from hypothesis, not per-triple.')
    log('  FactSpotter has_added: always False (recall-only metric).')

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(build_summary(summary_metrics), f, indent=2, ensure_ascii=False)

    print(f'\nLog saved → {log_file}')
    print(f'Summary saved → {summary_file}')


if __name__ == '__main__':
    run()
