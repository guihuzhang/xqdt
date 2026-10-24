#!/usr/bin/env python3
"""Score saved verifier predictions against the released fixed test annotations."""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def align_fixed_gold(predictions, gold, source):
    by_id = {row['sample_id']: row for row in predictions}
    if len(predictions) != len(gold) or set(by_id) != set(range(len(gold))):
        raise ValueError(f'{source}: incomplete or duplicate sample IDs')
    aligned = []
    changed = 0
    for sample_id, annotation in enumerate(gold):
        original = by_id[sample_id]
        if original.get('input_text') != annotation.get('input_text') or original.get('input_triples') != annotation.get('triples'):
            raise ValueError(f'{source}: input mismatch at sample_id {sample_id}')
        updates = {
            'label': annotation.get('label'),
            'gt_extra_triples': annotation.get('extra_contents', []) or [],
            'gt_missing_triples': annotation.get('missing_triples', []) or [],
            'gt_incorrect_mappings': annotation.get('incorrect_mappings', []) or [],
        }
        changed += any(original.get(key) != value for key, value in updates.items())
        aligned.append(dict(original, **updates))
    return aligned, changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['webnlg', 'e2e19'], required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--prediction-dir', type=Path,
                        help='Score test_predictions_*.json in this directory instead of the bundled model groups.')
    parser.add_argument('--similarity', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    workflow = root / ('verifier_train_eval_' + args.dataset)
    test_name = 'test_all_samples_corrected.json' if args.dataset == 'webnlg' else 'test_all_samples.json'
    gold_path = workflow / 'data' / test_name
    gold = json.loads(gold_path.read_text())
    module_name = ('verifier_train_eval_webnlg.verifier_eval.evaluate_json_embedding'
                   if args.dataset == 'webnlg' else
                   'verifier_train_eval_e2e19.verifier_eval.evaluate_error_detection_e2e_gemma')
    scorer = importlib.import_module(module_name)
    embedder = None
    if args.similarity:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer('BAAI/bge-large-en-v1.5')
    groups = ({args.prediction_dir.resolve(): None} if args.prediction_dir else
              {workflow / 'xqdt_outputs': 12, workflow / 'prompt_baselines': 5})
    results = []
    for group, expected in groups.items():
        files = sorted(path for path in group.glob('test_predictions_*.json')
                       if not path.stem.endswith('_evaluated'))
        if not files:
            raise FileNotFoundError(f'No prediction files found in {group}')
        if expected is not None and len(files) != expected:
            raise ValueError(f'{group}: expected {expected} prediction files, found {len(files)}')
        for path in files:
            predictions = json.loads(path.read_text())
            aligned, changed = align_fixed_gold(predictions, gold, path.name)
            metrics = []
            for sample in aligned:
                metrics.append(scorer.evaluate_sample(
                    scorer.extract_predicted_triples(sample), scorer.extract_gold_triples(sample),
                    embedder, sample.get('label') == 'positive'))
            aggregated = scorer.compute_global_metrics(metrics)
            if not args.similarity:
                for values in aggregated.values():
                    if isinstance(values, dict):
                        values.pop('similarity', None)
            try:
                file_name = str(path.relative_to(root))
            except ValueError:
                file_name = path.name
            result = {'file': file_name, 'samples': len(gold),
                      'updated_gold_rows': changed, 'metrics': aggregated}
            results.append(result)
            print(f'{path.name}: {len(gold)} samples, {changed} updated gold rows', flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / (args.dataset + '_scores.json')
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    output.write_text(json.dumps({'dataset': args.dataset, 'gold': str(gold_path.relative_to(root)),
                                 'similarity': args.similarity, 'results': results,
                                 'created_utc': datetime.now(timezone.utc).isoformat(),
                                 'python': sys.version.split()[0]}, indent=2) + '\n')
    lines = ['| Model output | Extra F1 | Omitted F1 | Incorrect F1 | Macro F1 |',
             '| --- | ---: | ---: | ---: | ---: |']
    for result in results:
        metrics = result['metrics']
        values = [metrics[key]['triple_level']['f1'] * 100 for key in ['extra', 'missing', 'incorrect', 'macro_average']]
        lines.append('| ' + Path(result['file']).stem + ' | ' + ' | '.join(f'{value:.1f}' for value in values) + ' |')
    (args.output_dir / (args.dataset + '_scores.md')).write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
