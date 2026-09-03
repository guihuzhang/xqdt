"""Lightweight E2E correlation helpers without model-training dependencies.

IMPORTANT: reproducing the paper's published Table 19 numbers requires the
`drop_gt100_all+A+C+D+E+F` worker-filtering strategy below (verified against
dt-explainable-metric/data/err_pred_e2e19/llm_inference/analyze_worker_filtering_full_metrics.py
and against the published Table 19 values). Passing raw, unfiltered
`human_quality_scores` (as stored in *_results.json) into compute_correlations
silently produces numbers that do NOT match the paper (e.g. Qwen3-8B system-level
Pearson r comes out ~15% instead of the published 79.8%). Always call
rebuild_human_quality_scores() first, using human_ratings/converted.json, and
pass the rebuilt results into compute_correlations().
"""

import random
import warnings
from collections import defaultdict
from typing import Dict, List

import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr

try:
    from .worker_filter import load_excluded_workers
except ImportError:
    from worker_filter import load_excluded_workers

# Worker groups identified in Table 18 (annotation quality issues) that must be
# excluded before computing correlations, per drop_gt100_all+A+C+D+E+F:
#   A: keyboard input error (repeated "1" key)
#   C: coarse/fine label mismatch
#   D: fixed-value raters (always the same score)
#   E: ceiling-effect raters (always ~100)
#   F: wrong scale (1-4 instead of 0-100)
# Group B (out-of-range scores) is instead handled globally via the ">100" drop
# below, not via worker exclusion.
EXCLUDED_WORKERS = load_excluded_workers()


def rebuild_human_quality_scores(results: List[Dict], converted: Dict) -> List[Dict]:
    """Re-derive `human_quality_scores` per (mr_id, sys_name) from raw annotations
    in converted.json, dropping excluded workers and any individual score > 100.

    This matches Table 20's "5,515 valid ratings remain from 6,012 (8.3% removed)"
    and is required to reproduce the published Table 19 correlations.
    """
    rebuilt = []
    for row in results:
        mr_id, sys_name = str(row['mr_id']), row['sys_name']
        annotations = (
            converted.get(mr_id, {}).get('systems', {}).get(sys_name, {}).get('quality', [])
        )
        scores = []
        for annotation in annotations:
            if str(annotation['worker_id']) in EXCLUDED_WORKERS:
                continue
            score = annotation['fine_score']
            if score > 100:
                continue
            scores.append(score)
        new_row = dict(row)
        new_row['human_quality_scores'] = scores
        rebuilt.append(new_row)
    return rebuilt


def compute_correlations_for_metric(results: List[Dict], model_key: str = 'f1') -> Dict:
    """Single-metric (precision|recall|f1) text- and system-level correlation
    against mean human fine_score, matching
    dt-explainable-metric/data/err_pred_e2e19/llm_inference/analyze_worker_filtering_full_metrics.py
    exactly (verified against the paper's published Table 19 numbers). Use this
    rather than compute_correlations() when reproducing Table 19-style results:
    the joint precision/recall/f1 grouping in compute_correlations() does not
    reproduce the published numbers.

    `results` must already be filtered via rebuild_human_quality_scores().
    """
    mr_data: Dict = defaultdict(lambda: {'metric': [], 'qual': []})
    sys_model: Dict = defaultdict(list)
    sys_human: Dict = defaultdict(list)

    for row in results:
        scores = row['human_quality_scores']
        if not scores:
            continue
        qual = np.mean(scores) / 100.0
        metric = row.get(model_key, float('nan'))
        if not np.isfinite(metric):
            continue
        mid, sys_name = row['mr_id'], row['sys_name']
        mr_data[mid]['metric'].append(metric)
        mr_data[mid]['qual'].append(qual)
        sys_model[sys_name].append(metric)
        sys_human[sys_name].append(qual)

    mr_ids = [mid for mid, data in mr_data.items() if len(data['metric']) >= 2]
    tp, ts, tk = [], [], []
    for mid in mr_ids:
        x = np.asarray(mr_data[mid]['metric'])
        y = np.asarray(mr_data[mid]['qual'])
        tp.append(_safe_corr(pearsonr, x, y))
        ts.append(_safe_corr(spearmanr, x, y))
        tk.append(_safe_corr(kendalltau, x, y))
    text_level = {
        'n_mr': len(mr_ids),
        'pearson': _safe_mean(tp), 'spearman': _safe_mean(ts), 'kendall': _safe_mean(tk),
    }

    sys_names = sorted(sys_model.keys())
    sx = np.asarray([np.mean(sys_model[s]) for s in sys_names])
    sy = np.asarray([np.mean(sys_human[s]) for s in sys_names])
    system_level = {
        'n_systems': len(sys_names),
        'pearson': _safe_corr(pearsonr, sx, sy),
        'spearman': _safe_corr(spearmanr, sx, sy),
        'kendall': _safe_corr(kendalltau, sx, sy),
    }
    return {'text_level': text_level, 'system_level': system_level}


def _bootstrap_ci(values: list, ci: float = 0.95) -> Dict:
    """Compute bootstrap mean and confidence interval."""
    if not values:
        return {'mean': 0.0, 'ci_low': 0.0, 'ci_high': 0.0, 'n_boot': 0}
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {'mean': 0.0, 'ci_low': 0.0, 'ci_high': 0.0, 'n_boot': 0}
    lo = (1 - ci) / 2
    hi = 1 - lo
    return {
        'mean': float(np.mean(arr)),
        'ci_low': float(np.percentile(arr, lo * 100)),
        'ci_high': float(np.percentile(arr, hi * 100)),
        'n_boot': int(arr.size),
    }


def _safe_corr(corr_fn, x: np.ndarray, y: np.ndarray) -> float:
    """Return a correlation value, or NaN when undefined (e.g. zero variance).

    NaN (not 0.0) is the correct fallback here: _safe_mean() below filters NaNs
    out of an average, so a degenerate per-MR correlation is excluded rather
    than counted as a real zero. Silently returning 0.0 instead systematically
    biases text-level averages downward (previously caused computed Table 19
    text-level numbers to come out ~4-5 points below the published values).
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            value = corr_fn(x, y)[0]
    except Exception:
        return float('nan')
    if value is None or not np.isfinite(value):
        return float('nan')
    return float(value)


def _safe_mean(values: list) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else 0.0


def compute_correlations(results: List[Dict], model_name: str, n_bootstrap: int = 1000) -> Dict:
    """Compute the paper's sample-, text-, and system-level E2E correlations.

    `results` must already be filtered via rebuild_human_quality_scores() —
    this function does not filter workers itself. Passing raw *_results.json
    entries directly will not reproduce the paper's published Table 19 numbers.
    """
    print(f'\nCorrelations: {model_name}')
    mr_data = defaultdict(lambda: {'prec': [], 'rec': [], 'f1': [], 'qual': []})
    sys_model = defaultdict(lambda: {'prec': [], 'rec': [], 'f1': []})
    sys_human = defaultdict(list)
    all_scores = {'prec': [], 'rec': [], 'f1': [], 'qual': []}

    for row in results:
        if not row['human_quality_scores']:
            continue
        mid, system = row['mr_id'], row['sys_name']
        quality = np.mean(row['human_quality_scores']) / 100.0
        for source, target in (('precision', 'prec'), ('recall', 'rec'), ('f1', 'f1')):
            mr_data[mid][target].append(row[source])
            sys_model[system][target].append(row[source])
            all_scores[target].append(row[source])
        mr_data[mid]['qual'].append(quality)
        sys_human[system].append(quality)
        all_scores['qual'].append(quality)

    arrays = {key: np.asarray(value) for key, value in all_scores.items()}
    methods = {'pearson': pearsonr, 'spearman': spearmanr, 'kendall': kendalltau}

    def correlations(prec, rec, f1, qual):
        return {
            name: {
                'precision': _safe_corr(fn, prec, qual),
                'recall': _safe_corr(fn, rec, qual),
                'f1': _safe_corr(fn, f1, qual),
            }
            for name, fn in methods.items()
        }

    sample_point = correlations(arrays['prec'], arrays['rec'], arrays['f1'], arrays['qual'])
    sample_boot = {name: {metric: [] for metric in ('precision', 'recall', 'f1')} for name in methods}
    n = len(arrays['qual'])
    for _ in range(n_bootstrap):
        idx = np.random.randint(0, n, size=n)
        boot = correlations(arrays['prec'][idx], arrays['rec'][idx], arrays['f1'][idx], arrays['qual'][idx])
        for name in methods:
            for metric in sample_boot[name]:
                sample_boot[name][metric].append(boot[name][metric])
    sample_corr = {
        'n': n,
        'point': sample_point,
        'bootstrap': {
            name: {metric: _bootstrap_ci(values) for metric, values in metrics.items()}
            for name, metrics in sample_boot.items()
        },
    }

    mr_ids = [mid for mid, data in mr_data.items() if len(data['prec']) >= 2]

    def text_point(ids):
        collected = {name: {metric: [] for metric in ('precision', 'recall', 'f1')} for name in methods}
        for mid in ids:
            data = mr_data[mid]
            point = correlations(
                np.asarray(data['prec']), np.asarray(data['rec']),
                np.asarray(data['f1']), np.asarray(data['qual']))
            for name in methods:
                for metric in collected[name]:
                    collected[name][metric].append(point[name][metric])
        return {
            name: {metric: _safe_mean(values) for metric, values in metrics.items()}
            for name, metrics in collected.items()
        }

    text_value = text_point(mr_ids)
    text_boot = {name: {metric: [] for metric in ('precision', 'recall', 'f1')} for name in methods}
    for _ in range(n_bootstrap):
        value = text_point(random.choices(mr_ids, k=len(mr_ids)))
        for name in methods:
            for metric in text_boot[name]:
                text_boot[name][metric].append(value[name][metric])
    text_corr = {
        'n_mr': len(mr_ids),
        'point': text_value,
        'bootstrap': {
            name: {metric: _bootstrap_ci(values) for metric, values in metrics.items()}
            for name, metrics in text_boot.items()
        },
    }

    systems = sorted(sys_model)
    sys_arrays = {
        metric: np.asarray([np.mean(sys_model[system][metric]) for system in systems])
        for metric in ('prec', 'rec', 'f1')
    }
    sys_quality = np.asarray([np.mean(sys_human[system]) for system in systems])
    sys_point = correlations(sys_arrays['prec'], sys_arrays['rec'], sys_arrays['f1'], sys_quality)
    sys_boot = {name: {metric: [] for metric in ('precision', 'recall', 'f1')} for name in methods}
    for _ in range(n_bootstrap):
        sampled = random.choices(range(len(systems)), k=len(systems))
        value = correlations(
            sys_arrays['prec'][sampled], sys_arrays['rec'][sampled],
            sys_arrays['f1'][sampled], sys_quality[sampled])
        for name in methods:
            for metric in sys_boot[name]:
                sys_boot[name][metric].append(value[name][metric])
    system_corr = {
        'n_systems': len(systems),
        'systems': systems,
        'point': sys_point,
        'bootstrap': {
            name: {metric: _bootstrap_ci(values) for metric, values in metrics.items()}
            for name, metrics in sys_boot.items()
        },
    }

    for level, values in (('Sample', sample_corr), ('Text', text_corr), ('System', system_corr)):
        print(f'{level}-level:')
        for name in methods:
            point = values['point'][name]
            print(f"  {name:<8} P={point['precision']:.4f} R={point['recall']:.4f} F1={point['f1']:.4f}")

    return {
        'model': model_name,
        'sample_level': sample_corr,
        'text_level': text_corr,
        'system_level': system_corr,
    }
