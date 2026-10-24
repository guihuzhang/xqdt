#!/usr/bin/env python3
"""Recompute T2D tables from saved triples using one evaluation environment."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from functools import lru_cache
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import reconstruct_compact_outputs as compact
import table_export_utils as tables
import validate_maintext_end_to_end as validation

HERE = Path(__file__).resolve().parent
METRICS = ('Ent_type', 'Partial', 'Strict', 'Exact')
FIELDS = ('Precision', 'Recall', 'F1')


def load_scorer():
    path = HERE / 'eval/eval_webnlg_t2g/Evaluation_script.py'
    spec = importlib.util.spec_from_file_location('t2d_scorer', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.tqdm = None
    module.evaluaterefcand = lru_cache(maxsize=32768)(module.evaluaterefcand.__wrapped__)
    tokenize = module.nltk.word_tokenize

    @lru_cache(maxsize=32768)
    def cached_tokens(text):
        return tuple(tokenize(text))

    module.nltk = SimpleNamespace(word_tokenize=lambda text: list(cached_tokens(text)))
    return module


def score_partition(task):
    dataset, baseline, models, methods, output_dir = task
    scorer = load_scorer()
    wrapper = validation.load_eval_module()
    reference_path = wrapper.REFERENCE_XMLS[dataset]
    _, references = scorer.getRefs(reference_path)
    reference_ids = [entry.attrib['eid'] for entry in ET.parse(reference_path).getroot().find('entries')]
    assert len(reference_ids) == len(references)

    @lru_cache(maxsize=None)
    def score_example(reference, candidate):
        pair_scores, tag_scores = scorer.calculateAllScores([list(reference)], [list(candidate)])
        selected, _, _ = scorer.select_best_alignment(pair_scores[0], tag_scores[0])
        return tuple(tuple(row[metric.lower()][field.lower()] for row in selected)
                     for metric in METRICS for field in FIELDS)

    def score_state(sample_ids, state, model, method, round_id):
        payload = compact.payload(sample_ids, state, dataset=dataset, baseline=baseline,
                                  model=model, prompt_variant=compact.PROMPT_VARIANTS[method])
        predictions = wrapper.extract_predictions(payload)
        missing = set(reference_ids) - predictions.keys()
        if missing:
            raise ValueError(f'{dataset}/{baseline}: missing predictions for {sorted(missing)[:5]}')
        candidates = scorer.prepareList([predictions[sample_id] for sample_id in reference_ids])
        collected = [[] for _ in range(len(METRICS) * len(FIELDS))]
        for reference, candidate in zip(references, candidates):
            values = score_example(tuple(reference), tuple(candidate))
            for target, source in zip(collected, values):
                target.extend(source)
        totals = {}
        for metric_index, metric in enumerate(METRICS):
            totals[metric] = {field: statistics.mean(collected[metric_index * len(FIELDS) + field_index])
                             for field_index, field in enumerate(FIELDS)}
        return dict(dataset=dataset, baseline=baseline, model=model if round_id else None,
                    method=method if round_id else 'baseline', round=round_id,
                    num_samples=len(reference_ids), total_scores=totals)

    results = []
    for model in models:
        for method in methods:
            sample_ids, states = validation.reconstruct_states(method, model, dataset, baseline)
            if not results:
                results.append(score_state(sample_ids, states[0], model, method, 0))
            for round_id in (1, 2, 3):
                results.append(score_state(sample_ids, states[round_id], model, method, round_id))
            print(f'{dataset}/{baseline}: {model}/{method} complete ({len(results)} runs)', flush=True)
    path = output_dir / f'{dataset}_{baseline}.json'
    path.write_text(json.dumps(dict(runs=results, unique_sample_states=score_example.cache_info().currsize), indent=2) + '\n')
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--datasets', nargs='+', choices=('genwiki', 'webnlg'), default=['genwiki', 'webnlg'])
    parser.add_argument('--baselines', nargs='+', choices=('bt5', 'regen'), default=['bt5', 'regen'])
    parser.add_argument('--models', nargs='+', choices=tables.APPENDIX_MODELS, default=tables.APPENDIX_MODELS)
    parser.add_argument('--methods', nargs='+', choices=compact.PROMPT_VARIANTS, default=list(compact.PROMPT_VARIANTS))
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    packages = {name: importlib.metadata.version(name) for name in
                ('nervaluate', 'nltk', 'numpy', 'scipy', 'scikit-learn', 'beautifulsoup4', 'lxml', 'regex')}
    if packages['nervaluate'] != '0.2.0' or packages['nltk'] != '3.9.4':
        raise RuntimeError('Install requirements-repair_t2d-eval-py311.txt: nervaluate 0.2.0 and nltk 3.9.4 are required.')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    tasks = [(dataset, baseline, args.models, args.methods, args.output_dir)
             for dataset in args.datasets for baseline in args.baselines]
    if args.workers == 1:
        partitions = [score_partition(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as executor:
            partitions = list(executor.map(score_partition, tasks))
    runs = [run for partition in partitions for run in partition]
    expected = len(tasks) * (1 + len(args.models) * len(args.methods) * 3)
    assert len(runs) == expected
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=HERE, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    inputs = sorted(set([*list((HERE / 't2d_repair_input').rglob('*.json')),
                         *list((HERE / 'results/repair_outputs').glob('*.jsonl')),
                         *list((HERE / 'eval/data').glob('*refs.xml'))]))
    metadata = dict(started_utc=started, finished_utc=datetime.now(timezone.utc).isoformat(),
                    python=sys.version.split()[0], packages=packages, code_commit=commit,
                    command=['python', 'repair_t2d/rescore_saved_outputs.py', '--output-dir', '<new-directory>',
                             '--workers', str(args.workers), '--datasets', *args.datasets,
                             '--baselines', *args.baselines, '--models', *args.models, '--methods', *args.methods],
                    inputs=[dict(path=str(path.relative_to(HERE)), bytes=path.stat().st_size) for path in inputs],
                    aggregation='Mean of matched-triple P/R/F1 after the bundled Hungarian alignment; unchanged evaluator.',
                    optimization='Reuse identical normalized reference/candidate sample states within each partition.',
                    run_count=len(runs))
    (args.output_dir / 'scores.json').write_text(json.dumps(dict(metadata=metadata, runs=runs), indent=2) + '\n')
    print(f'Completed {len(runs)} runs: {args.output_dir / "scores.json"}', flush=True)


if __name__ == '__main__':
    main()
