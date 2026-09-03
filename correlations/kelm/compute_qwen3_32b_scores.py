#!/usr/bin/env python3
"""Recompute the Qwen3-32B KELM prompted-baseline scores."""

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np


def parse_args():
    base = Path(__file__).resolve().parent / "prompted_llm_results"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=base / "Qwen3-32B-Prompt_predictions.jsonl",
    )
    parser.add_argument(
        "--prediction-review",
        type=Path,
        default=base / "Qwen3-32B-Prompt_prediction_review.csv",
    )
    parser.add_argument(
        "--missed-errors",
        type=Path,
        default=base / "Qwen3-32B-Prompt_missed_errors.csv",
    )
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2023)
    return parser.parse_args()


def normalize_sample_id(value):
    match = re.search(r"Id\d+", str(value))
    if not match:
        raise ValueError(f"Cannot extract sample ID from {value!r}")
    return match.group()


def read_sample_ids(path):
    sample_ids = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                sample_ids.append(
                    normalize_sample_id(json.loads(line)["sample_id"])
                )
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Prediction file contains duplicate sample IDs")
    return sample_ids


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def collect_counts(prediction_review, missed_errors, strict=False):
    counts = defaultdict(Counter)
    for row in read_csv(prediction_review):
        sample_id = normalize_sample_id(row["sample_id"])
        decision = row["decision_lenient"].strip().upper()
        if decision == "TP":
            counts[sample_id]["TP"] += 1
        elif decision == "FP":
            counts[sample_id]["FP"] += 1
        elif decision == "FN":
            counts[sample_id]["FN"] += 1
            if strict:
                counts[sample_id]["FP"] += 1
        else:
            raise ValueError(f"Unexpected prediction decision: {decision!r}")

    for row in read_csv(missed_errors):
        decision = row["decision"].strip().upper()
        if decision != "FN":
            raise ValueError(f"Unexpected missed-error decision: {decision!r}")
        counts[normalize_sample_id(row["sample_id"])]["FN"] += 1
    return counts


def sample_prf(counts):
    true_positive = counts["TP"]
    false_positive = counts["FP"]
    false_negative = counts["FN"]
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else (1.0 if false_negative == 0 else 0.0)
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else (1.0 if false_positive == 0 else 0.0)
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return precision, recall, f1


def bootstrap(sample_ids, counts, n_bootstrap, seed):
    sample_scores = np.asarray([sample_prf(counts[sid]) for sid in sample_ids])
    rng = random.Random(seed)
    scores = []
    for _ in range(n_bootstrap):
        indices = [rng.randrange(len(sample_ids)) for _ in sample_ids]
        scores.append(sample_scores[indices].mean(axis=0))
    scores = np.asarray(scores)
    lower, upper = np.percentile(scores[:, 2], [2.5, 97.5])
    return scores.mean(axis=0), lower, upper


def format_percent(value):
    return Decimal(f"{value * 100:.2f}").quantize(
        Decimal("0.1"), rounding=ROUND_HALF_UP
    )


def print_result(label, sample_ids, counts, n_bootstrap, seed):
    means, lower, upper = bootstrap(sample_ids, counts, n_bootstrap, seed)
    print(
        f"{label}: P={format_percent(means[0])}  "
        f"R={format_percent(means[1])}  "
        f"F1={format_percent(means[2])}  "
        f"F1 95% CI=[{format_percent(lower)}, {format_percent(upper)}]"
    )


def main():
    args = parse_args()
    if args.n_bootstrap <= 0:
        raise ValueError("--n-bootstrap must be positive")
    sample_ids = read_sample_ids(args.predictions)
    print(f"Samples: {len(sample_ids)}")
    print(f"Bootstrap iterations: {args.n_bootstrap}")
    print(f"Seed: {args.seed}")
    print_result(
        "Lenient",
        sample_ids,
        collect_counts(args.prediction_review, args.missed_errors),
        args.n_bootstrap,
        args.seed,
    )
    print_result(
        "Strict sensitivity",
        sample_ids,
        collect_counts(
            args.prediction_review, args.missed_errors, strict=True
        ),
        args.n_bootstrap,
        args.seed,
    )


if __name__ == "__main__":
    main()
