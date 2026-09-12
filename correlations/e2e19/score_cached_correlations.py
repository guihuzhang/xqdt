#!/usr/bin/env python3
"""Recompute E2E correlations from cached scores and human ratings, without inference."""

import argparse
import json
import math
from pathlib import Path

from baselines.ref_baseline_eval_e2e19 import (
    add_human_quality,
    compute_correlations,
    load_all_samples,
)
from table19_e2e_in_vs_cross_domain import MODEL_PAIRS


SCRIPT_DIR = Path(__file__).resolve().parent
BASELINE_METRICS = {
    "nli_prec": "NLI-P",
    "nli_rec": "NLI-R",
    "nli_f1": "NLI-F1",
    "monolr_prec": "MonoLR-P",
    "monolr_rec": "MonoLR-R",
    "monolr_f1": "MonoLR-F1",
    "factspotter_electra": "FactSpotter",
    "questeval_prec": "Data-QuestEval-P",
    "questeval_rec": "Data-QuestEval-R",
    "questeval_f1": "Data-QuestEval-F1",
}
REFERENCE_METRICS = {
    "bleu": "BLEU",
    "meteor": "METEOR",
    "parent_f1": "PARENT-F1",
    "bertscore_f1": "BERTScore-F1",
    "bartscore": "BARTScore",
    "bleurt": "BLEURT",
}


def load_json(path):
    if not path.is_file():
        raise FileNotFoundError(f"Required input not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def score_row(samples, metric_key, name):
    if not any(metric_key in sample for sample in samples):
        raise ValueError(f"No cached values for {name}")
    text, system, count = compute_correlations(samples, metric_key)
    if count == 0:
        raise ValueError(f"No valid evaluation pairs for {name}")
    return {
        "model": name,
        "n_valid_pairs": count,
        "system": dict(zip(("pearson", "spearman", "kendall"), system)),
        "text": dict(zip(("pearson", "spearman", "kendall"), text)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "correlation_results")
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    converted = load_json(data_dir / "human_ratings/converted.json")
    samples = load_all_samples(converted)
    excluded = {
        line.strip()
        for line in (data_dir / "human_ratings/excluded_workers.txt").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    add_human_quality(samples, converted, excluded_workers=excluded)
    sample_keys = {f'{sample["mr_id"]}||{sample["sys_name"]}' for sample in samples}
    rows = []
    for filename, metrics in (
        ("ref_baseline_scores_e2e19.json", REFERENCE_METRICS),
        ("baseline_scores_e2e19.json", BASELINE_METRICS),
    ):
        cache = load_json(data_dir / "baselines" / filename)
        if set(cache) != sample_keys:
            raise ValueError(f"Sample IDs do not match human inputs: {filename}")
        scored = []
        for sample in samples:
            entry = cache[f'{sample["mr_id"]}||{sample["sys_name"]}']
            values = {key: entry[key] for key in metrics if key in entry}
            scored.append(dict(sample, **values))
        rows.extend(score_row(scored, key, name) for key, name in metrics.items())
    for name, _, filename in MODEL_PAIRS:
        predictions = load_json(data_dir / "xqdt_results_repaired" / filename)
        keyed = {f'{row["mr_id"]}||{row["sys_name"]}': row for row in predictions}
        if len(keyed) != len(predictions) or set(keyed) != sample_keys:
            raise ValueError(f"Sample IDs do not match human inputs: {filename}")
        scored = []
        for sample in samples:
            prediction = keyed[f'{sample["mr_id"]}||{sample["sys_name"]}']
            scored.append(dict(sample, **{key: prediction[key] for key in ("precision", "recall", "f1")}))
        rows.extend(score_row(scored, key, f"{name}-{suffix}") for key, suffix in
                    (("precision", "P"), ("recall", "R"), ("f1", "F1")))
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "n_input_pairs": len(samples),
        "n_pairs_with_human_quality": sum(sample["human_quality"] is not None for sample in samples),
        "rows": rows,
    }
    (output_dir / "correlations.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    lines = [
        "| Model | Valid pairs | System r | System rho | System tau | Text r | Text rho | Text tau |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        values = [row[level][metric] for level in ("system", "text")
                  for metric in ("pearson", "spearman", "kendall")]
        scores = " | ".join(f"{value * 100:.1f}" if math.isfinite(value) else "—" for value in values)
        lines.append(f'| {row["model"]} | {row["n_valid_pairs"]} | {scores} |')
    (output_dir / "correlations.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Results written to {output_dir}")


if __name__ == "__main__":
    main()
