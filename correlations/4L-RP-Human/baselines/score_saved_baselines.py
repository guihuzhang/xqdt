"""Recompute traditional 4L-RP baseline correlations from saved predictions."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr


def score(rows, recall_only=False, bootstrap=False):
    if len(rows) != 50 or len({row['id'] for row in rows}) != 50:
        raise ValueError('Expected 50 distinct English examples.')
    dimensions = ['recall'] if recall_only else ['precision', 'recall', 'f1']
    predictions = np.array([[row[key] for key in dimensions] for row in rows])
    targets = []
    for row in rows:
        recall = float(np.mean(row['human_recall_list']))
        if recall_only:
            targets.append([recall])
        else:
            precision = float(np.mean(row['human_precision_list']))
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            targets.append([precision, recall, f1])
    targets = np.array(targets)
    if not np.isfinite(predictions).all() or not np.isfinite(targets).all():
        raise ValueError('Non-finite predictions or human scores.')
    functions = {'pearson': pearsonr, 'spearman': spearmanr, 'kendall': kendalltau}

    def correlate(indices):
        return {name: {key: float(function(predictions[indices, column], targets[indices, column])[0])
                       for column, key in enumerate(dimensions)}
                for name, function in functions.items()}

    result = {'n_samples': len(rows), 'point': correlate(np.arange(len(rows)))}
    if bootstrap:
        generator = np.random.RandomState(42)
        replicates = [correlate(generator.choice(len(rows), size=len(rows), replace=True))
                      for _ in range(1000)]
        result['bootstrap_mean'] = {
            name: {key: float(np.mean([replicate[name][key] for replicate in replicates]))
                   for key in dimensions} for name in functions}
        result['bootstrap_percentile_95_ci'] = {
            name: {key: np.percentile([replicate[name][key] for replicate in replicates], [2.5, 97.5]).tolist()
                   for key in dimensions} for name in functions}
        result['bootstrap_seed'] = 42
        result['bootstrap_resamples'] = 1000
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=Path(__file__).resolve().parent / 'results')
    parser.add_argument('--output-dir', type=Path, default=None)
    args = parser.parse_args()
    summary = {}
    for name in ['monolr', 'nli', 'factspotter_electra', 'questeval']:
        path = args.results_dir / f'{name}_results.json'
        rows = json.loads(path.read_text())
        result = score(rows, recall_only=name == 'factspotter_electra', bootstrap=name == 'questeval')
        summary[name] = result
        values = result.get('bootstrap_mean', result['point'])['spearman']
        print(name, 'Spearman × 100:', ' / '.join(f'{value * 100:.1f}' for value in values.values()))
    output_dir = args.output_dir or args.results_dir / 'rescored'
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'baseline_correlations.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
