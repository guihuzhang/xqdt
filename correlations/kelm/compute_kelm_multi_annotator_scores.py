#!/usr/bin/env python3
"""Compute Table 8-style KELM scores from multiple annotations."""

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def parse_args():
    base = Path(__file__).resolve().parent / "prompted_llm_results/kelm_gold_review"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotator-1-manual",
        type=Path,
        default=base / "manual_review_annotator_1.csv",
    )
    parser.add_argument(
        "--annotator-1-automatic",
        type=Path,
        default=base / "automatic_matches_annotator_1.csv",
    )
    parser.add_argument(
        "--annotator-2",
        type=Path,
        default=base / "manual_review_annotator_2.csv",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path(__file__).resolve().parent
        / "prompted_llm_results/GPT-5.1_predictions.jsonl",
    )
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2023)
    return parser.parse_args()


def sample_id(value):
    match = re.search(r"Id\d+", str(value))
    if not match:
        raise ValueError(f"Cannot extract sample ID from {value!r}")
    return match.group(0)


def read_csv(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing annotation file: {path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_sample_ids(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing prediction file: {path}")
    ids = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                ids.append(sample_id(json.loads(line)["sample_id"]))
    if len(ids) != len(set(ids)):
        raise ValueError("Prediction file contains duplicate sample IDs")
    return ids


def collect_decisions(paths):
    counts = defaultdict(Counter)
    for path in paths:
        for row in read_csv(path):
            decision = row.get("human_decision", "").strip().upper()
            if decision in {"TP", "FP", "FN"}:
                counts[sample_id(row["sample_id"])][decision] += 1
            elif decision not in {"", "IGNORE"}:
                raise ValueError(
                    f"Unexpected decision {decision!r} in {path}: "
                    f"{row.get('review_id', 'unknown row')}"
                )
    return counts


def sample_prf(counts):
    tp, fp, fn = counts["TP"], counts["FP"], counts["FN"]
    if tp + fp:
        precision = tp / (tp + fp)
    else:
        precision = 1.0 if fn == 0 else 0.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return precision, recall, f1


def main():
    args = parse_args()
    if args.n_bootstrap <= 0:
        raise ValueError("--n-bootstrap must be positive")

    sample_ids = read_sample_ids(args.predictions)
    annotations = [
        collect_decisions(
            [args.annotator_1_automatic, args.annotator_1_manual]
        ),
        collect_decisions([args.annotator_2]),
    ]
    per_sample = [
        [sample_prf(annotation[sid]) for annotation in annotations]
        for sid in sample_ids
    ]

    rng = np.random.default_rng(args.seed)
    bootstrap_scores = []
    for _ in range(args.n_bootstrap):
        sampled_indices = rng.integers(0, len(sample_ids), size=len(sample_ids))
        scores = [
            per_sample[index][rng.integers(0, len(annotations))]
            for index in sampled_indices
        ]
        bootstrap_scores.append(np.mean(scores, axis=0))

    bootstrap_scores = np.asarray(bootstrap_scores)
    means = bootstrap_scores.mean(axis=0) * 100
    f1_lower, f1_upper = np.percentile(
        bootstrap_scores[:, 2], [2.5, 97.5]
    ) * 100

    print(f"Samples: {len(sample_ids)}")
    print(f"Annotators: {len(annotations)}")
    print(f"Bootstrap iterations: {args.n_bootstrap}")
    print(f"Seed: {args.seed}")
    print(
        f"P={means[0]:.1f}  R={means[1]:.1f}  F1={means[2]:.1f}  "
        f"F1 95% CI=[{f1_lower:.1f}, {f1_upper:.1f}]"
    )


if __name__ == "__main__":
    main()
