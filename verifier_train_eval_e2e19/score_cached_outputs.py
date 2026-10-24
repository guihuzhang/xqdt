"""Recompute E2E triple- and element-level metrics without model inference."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from verifier_eval import evaluate_error_detection_e2e_gemma as xqdt_scorer
from verifier_eval import evaluate_error_detection_e2e_prompt as prompt_scorer
from verifier_train_eval_webnlg.score_verifier_release import align_fixed_gold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument('--output-dir', type=Path, default=root / 'cached_scores')
    args = parser.parse_args()
    gold = json.loads((root / 'data/test_all_samples.json').read_text())
    summary = {}
    for folder, expected, scorer in [('xqdt_outputs', 12, xqdt_scorer),
                                     ('prompt_baselines', 5, prompt_scorer)]:
        files = sorted((root / folder).glob('test_predictions_*.json'))
        if len(files) != expected:
            raise ValueError(f'{folder}: expected {expected} prediction files, found {len(files)}')
        for path in files:
            rows = json.loads(path.read_text())
            if len(rows) != 1229 or len({row['sample_id'] for row in rows}) != 1229:
                raise ValueError(f'{path.name}: expected 1229 distinct samples')
            rows, changed = align_fixed_gold(rows, gold, path.name)
            metrics = [scorer.evaluate_sample(scorer.extract_predicted_triples(row),
                                             scorer.extract_gold_triples(row),
                                             is_positive=row.get('label') == 'positive') for row in rows]
            global_metrics = scorer.compute_global_metrics(metrics)
            for values in global_metrics.values():
                values.pop('similarity', None)
            paper_row = [round(global_metrics[key]['triple_level']['f1'] * 100, 1)
                         for key in ['extra', 'missing', 'incorrect', 'macro_average']]
            summary[path.name] = {'samples': len(rows), 'updated_gold_rows': changed,
                                  'global_metrics': global_metrics,
                                  'paper_row': paper_row}
            print(path.name, paper_row)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
