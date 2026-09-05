import argparse
import json
import random
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import ConstantInputWarning, kendalltau, pearsonr, spearmanr


CORRELATIONS = (pearsonr, spearmanr, kendalltau)
CORRELATION_NAMES = (
    "system_pearson",
    "system_spearman",
    "system_kendall",
    "text_pearson",
    "text_spearman",
    "text_kendall",
)


WEBLG17_METRICS = (
    # Ordered (paper label, metric_results directory) pairs for Appendix Table 18.
    ("BLEU", "bleu"),
    ("METEOR", "meteor"),
    ("PARENT", "parent_f1"),
    ("BERTScore", "bert-f1"),
    ("BARTScore", "bart"),
    ("BLEURT", "bleurt"),
    ("Data-QuestEval-F1", "questeval"),
    ("FactSpotter", "fact-spotter"),
    ("NLI-F1", "nli_based_f1"),
    ("MonoLR-F1", "monolr-f1"),
    ("Gemma3-270M-F1", "dt_gemma3_270_16620_f1"),
    ("Gemma3-1B-F1", "dt_gemma3_1b_6648_f1"),
    ("Gemma3-4B-F1", "dt_gemma3_4b_6648_f1"),
    ("Gemma3-12B-F1", "dt_gemma3_12b_4432_f1"),
    ("Qwen3-0.6B-F1", "dt_qwen3_0.6b_8864_f1"),
    ("Qwen3-1.7B-F1", "dt_qwen3_1.7b_6648_f1"),
    ("Qwen3-4B-F1", "dt_qwen3_4b_4432_f1"),
    ("Qwen3-8B-F1", "dt_qwen3_8b_7479_f1"),
    ("Qwen3-14B-F1", "dt_qwen3_14b_4432_f1"),
    ("Llama3.2-1B-F1", "dt_llama3.2_1b_8864_f1"),
    ("Llama3.2-3B-F1", "dt_llama3.2_3b_5540_f1"),
    ("Llama3.1-8B-F1", "dt_llama3.1_8b_7756_f1"),
)


def parse_args():
    experiment_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Recompute the WebNLG 2017 semantic-adequacy correlations."
    )
    parser.add_argument("--experiment-dir", type=Path, default=experiment_dir)
    parser.add_argument("--output-dir", type=Path, default=experiment_dir / "results")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=10)
    return parser.parse_args()


def read_scores(path):
    return np.asarray([float(value) for value in path.read_text().splitlines()])


def load_matrices(experiment_dir, metric_directory):
    metric_dir = experiment_dir / "metric_results" / metric_directory
    human_dir = experiment_dir / "human-annotations" / "semantics"
    systems = sorted(path.name for path in metric_dir.glob("*.txt"))
    if len(systems) != 9:
        raise ValueError(f"Expected 9 systems in {metric_dir}, found {len(systems)}")
    metric_scores = np.stack([read_scores(metric_dir / system) for system in systems])
    human_scores = np.stack([read_scores(human_dir / system) for system in systems])
    if metric_scores.shape != (9, 223) or human_scores.shape != (9, 223):
        raise ValueError(
            f"Expected score matrices of shape (9, 223), found "
            f"{metric_scores.shape} and {human_scores.shape}"
        )
    return metric_scores, human_scores


def significant_statistic(function, left, right):
    result = function(left, right)
    if np.isnan(result.statistic) or result.pvalue >= 0.05:
        return None
    return float(result.statistic)


def system_level(metric_scores, human_scores, bootstrap_indices):
    results = []
    for function in CORRELATIONS:
        coefficients = []
        for indices in bootstrap_indices:
            coefficient = significant_statistic(
                function,
                metric_scores[:, indices].mean(axis=1),
                human_scores[:, indices].mean(axis=1),
            )
            if coefficient is not None:
                coefficients.append(coefficient)
        results.append(float(np.mean(coefficients)) if coefficients else None)
    return results


def text_level(metric_scores, human_scores, bootstrap_indices):
    results = []
    for function in CORRELATIONS:
        per_example = np.full(metric_scores.shape[1], np.nan)
        for example_index in range(metric_scores.shape[1]):
            coefficient = significant_statistic(
                function,
                metric_scores[:, example_index],
                human_scores[:, example_index],
            )
            if coefficient is not None:
                per_example[example_index] = coefficient
        replicate_means = []
        for indices in bootstrap_indices:
            selected = per_example[indices]
            selected = selected[~np.isnan(selected)]
            if selected.size:
                replicate_means.append(float(selected.mean()))
        results.append(float(np.mean(replicate_means)) if replicate_means else None)
    return results


def rounded_percentages(values):
    return [None if value is None else round(value * 100, 1) for value in values]


def markdown_table(rows):
    header = (
        "| Model | System r | System rho | System tau | "
        "Text r | Text rho | Text tau |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    lines = [header]
    for row in rows:
        values = ["--" if value is None else f"{value:.1f}" for value in row["rounded"]]
        lines.append(f"| {row['label']} | " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    generator = random.Random(args.seed)
    population = list(range(223))
    bootstrap_indices = np.asarray(
        [generator.choices(population, k=223) for _ in range(args.bootstrap_samples)]
    )
    rows = []
    warnings.filterwarnings("ignore", category=ConstantInputWarning)
    for label, directory in WEBLG17_METRICS:
        metric_scores, human_scores = load_matrices(args.experiment_dir, directory)
        raw_values = system_level(metric_scores, human_scores, bootstrap_indices)
        raw_values += text_level(metric_scores, human_scores, bootstrap_indices)
        rows.append(
            {
                "label": label,
                "directory": directory,
                "raw": dict(zip(CORRELATION_NAMES, raw_values)),
                "rounded": rounded_percentages(raw_values),
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": "WebNLG 2017",
        "human_dimension": "semantic adequacy",
        "systems": 9,
        "examples_per_system": 223,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "rows": rows,
    }
    (args.output_dir / "webnlg17_correlations.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    table = markdown_table(rows)
    (args.output_dir / "webnlg17_correlations.md").write_text(table)
    print(table, end="")


if __name__ == "__main__":
    main()
