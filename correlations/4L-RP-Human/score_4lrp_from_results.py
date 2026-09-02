#!/usr/bin/env python3
"""Score 4L-RP-Human correlations from saved XQDT result files.

This script does not run inference. It only reads saved `*_results.json` files,
computes Pearson / Spearman / Kendall / RMSE against the human annotations,
and writes `*_correlation.json` plus a `summary.json`.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "xqdt_results"


def safe_corr(corr_fn, x: np.ndarray, y: np.ndarray) -> float:
    """Return a finite correlation value, or 0.0 when undefined."""
    try:
        value = corr_fn(x, y)[0]
    except Exception:
        return 0.0
    if value is None or not np.isfinite(value):
        return 0.0
    return float(value)


def compute_correlations(results: List[Dict[str, Any]], model_name: str) -> Optional[Dict[str, Any]]:
    """Compute correlation metrics from saved sample-level scores."""
    model_precision = []
    model_recall = []
    model_f1 = []
    human_precision_avg = []
    human_recall_avg = []
    human_f1_avg = []

    for item in results:
        human_precision_list = item.get("human_precision_list", [])
        human_recall_list = item.get("human_recall_list", [])
        if not human_precision_list or not human_recall_list:
            continue

        precision = float(item["precision"])
        recall = float(item["recall"])
        f1 = float(item["f1"])

        human_precision = float(np.mean(human_precision_list))
        human_recall = float(np.mean(human_recall_list))
        if human_precision + human_recall > 0:
            human_f1 = (2 * human_precision * human_recall) / (human_precision + human_recall)
        else:
            human_f1 = 0.0

        model_precision.append(precision)
        model_recall.append(recall)
        model_f1.append(f1)
        human_precision_avg.append(human_precision)
        human_recall_avg.append(human_recall)
        human_f1_avg.append(human_f1)

    if not model_precision:
        return None

    model_precision = np.asarray(model_precision, dtype=float)
    model_recall = np.asarray(model_recall, dtype=float)
    model_f1 = np.asarray(model_f1, dtype=float)
    human_precision_avg = np.asarray(human_precision_avg, dtype=float)
    human_recall_avg = np.asarray(human_recall_avg, dtype=float)
    human_f1_avg = np.asarray(human_f1_avg, dtype=float)

    pearson = {
        "precision": safe_corr(pearsonr, model_precision, human_precision_avg),
        "recall": safe_corr(pearsonr, model_recall, human_recall_avg),
        "f1": safe_corr(pearsonr, model_f1, human_f1_avg),
    }
    spearman = {
        "precision": safe_corr(spearmanr, model_precision, human_precision_avg),
        "recall": safe_corr(spearmanr, model_recall, human_recall_avg),
        "f1": safe_corr(spearmanr, model_f1, human_f1_avg),
    }
    kendall = {
        "precision": safe_corr(kendalltau, model_precision, human_precision_avg),
        "recall": safe_corr(kendalltau, model_recall, human_recall_avg),
        "f1": safe_corr(kendalltau, model_f1, human_f1_avg),
    }
    rmse = {
        "precision": float(np.sqrt(np.mean((model_precision - human_precision_avg) ** 2))),
        "recall": float(np.sqrt(np.mean((model_recall - human_recall_avg) ** 2))),
        "f1": float(np.sqrt(np.mean((model_f1 - human_f1_avg) ** 2))),
    }

    return {
        "model": model_name,
        "n_samples": int(model_precision.size),
        "pearson": pearson,
        "spearman": spearman,
        "kendall": kendall,
        "rmse": rmse,
    }


def score_results_file(results_file: Path) -> Optional[Dict[str, Any]]:
    """Load one saved result file and compute its correlation summary."""
    with open(results_file, "r", encoding="utf-8") as f:
        results = json.load(f)
    model_name = results_file.stem.removesuffix("_results")
    return compute_correlations(results, model_name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing saved `*_results.json` files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for `*_correlation.json` and `summary.json`. Defaults to --results-dir.",
    )
    parser.add_argument(
        "--pattern",
        default="*_results.json",
        help="Glob pattern used to discover result files.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional log file path. Defaults to evaluation_<timestamp>.log in --output-dir.",
    )
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    output_dir = (args.output_dir or results_dir).resolve()

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    results_files = sorted(results_dir.glob(args.pattern))
    if not results_files:
        raise FileNotFoundError(
            f"No result files found in {results_dir} with pattern {args.pattern!r}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = args.log_file.resolve() if args.log_file else output_dir / (
        f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    summary = []
    log_lines = [
        "4L-RP-Human correlation scoring",
        f"results_dir: {results_dir}",
        f"output_dir: {output_dir}",
        f"pattern: {args.pattern}",
        "",
    ]

    for results_file in results_files:
        correlation = score_results_file(results_file)
        if correlation is None:
            message = f"Skipping {results_file.name}: no valid samples"
            print(message)
            log_lines.append(message)
            continue

        summary.append(correlation)
        output_file = output_dir / f"{correlation['model']}_correlation.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(correlation, f, indent=2, ensure_ascii=False)

        spearman = correlation["spearman"]
        pearson = correlation["pearson"]
        kendall = correlation["kendall"]
        rmse = correlation["rmse"]
        model_log = (
            f"{correlation['model']}: "
            f"Pearson P={pearson['precision']:.4f}, R={pearson['recall']:.4f}, F1={pearson['f1']:.4f} | "
            f"Spearman P={spearman['precision']:.4f}, R={spearman['recall']:.4f}, F1={spearman['f1']:.4f} | "
            f"Kendall P={kendall['precision']:.4f}, R={kendall['recall']:.4f}, F1={kendall['f1']:.4f} | "
            f"RMSE P={rmse['precision']:.4f}, R={rmse['recall']:.4f}, F1={rmse['f1']:.4f}"
        )
        print(model_log)
        log_lines.append(model_log)

    summary_file = output_dir / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    footer_lines = [
        "",
        f"Saved {len(summary)} correlation files to {output_dir}",
        f"Summary saved to {summary_file}",
        f"Log saved to {log_file}",
    ]
    for line in footer_lines:
        print(line)
    log_lines.extend(footer_lines)

    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")


if __name__ == "__main__":
    main()
