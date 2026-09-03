#!/usr/bin/env python3
"""Paired bootstrap for WebNLG-only -> E2E transfer versus joint training.

The two settings are evaluated on identical E2E (MR, system) pairs. Coarse-label
F1 and text-level correlations are bootstrapped by MR; system-level correlations
are bootstrapped separately by system. Every replicate uses the same sampled
units for the two settings and reports cross-domain minus joint-training.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shlex
import subprocess
import sys
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import scipy
from scipy.stats import kendalltau, pearsonr, spearmanr

try:
    from .worker_filter import load_excluded_workers
except ImportError:
    from worker_filter import load_excluded_workers

EXCLUDED_WORKERS = load_excluded_workers()
MISSING_LABELS = {"missing", "added missing"}
ADDED_LABELS = {"added", "added missing"}
TASKS = ("is_ok", "has_missing", "has_added")
CORRELATIONS: dict[str, Callable] = {
    "pearson": pearsonr,
    "spearman": spearmanr,
    "kendall": kendalltau,
}
MODEL_PAIRS = {
    "Qwen3-4B": {
        "joint": "correlations/e2e19/xqdt_results_repaired/qwen3_4b_8370_results.json",
        "cross": "verifier_train_eval_e2e19/cross_dataset_results/webnlg_to_e2e_human/qwen3_4b_4432_results.json",
    },
    "Qwen3-8B": {
        "joint": "correlations/e2e19/xqdt_results_repaired/qwen3_8b_8680_results.json",
        "cross": "verifier_train_eval_e2e19/cross_dataset_results/webnlg_to_e2e_human/qwen3_8b_7479_results.json",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Root of the XQDT repository.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_PAIRS),
        default=list(MODEL_PAIRS),
    )
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Parent directory for a newly timestamped result directory.",
    )
    parser.add_argument("--chunk-size", type=int, default=250)
    args = parser.parse_args()
    if args.n_bootstrap <= 0:
        parser.error("--n-bootstrap must be positive")
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")
    return args


def load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def stable_key(row: dict) -> tuple[str, str]:
    return str(row["mr_id"]), str(row["sys_name"])


def index_unique(rows: list[dict], path: Path) -> dict[tuple[str, str], dict]:
    indexed: dict[tuple[str, str], dict] = {}
    for row in rows:
        key = stable_key(row)
        if key in indexed:
            raise ValueError(f"Duplicate (mr_id, sys_name) key {key} in {path}")
        indexed[key] = row
    return indexed


def filtered_annotations(converted: dict, key: tuple[str, str]) -> list[dict]:
    mr_id, sys_name = key
    annotations = converted.get(mr_id, {}).get("systems", {}).get(sys_name, {}).get("quality", [])
    return [
        ann for ann in annotations
        if str(ann["worker_id"]) not in EXCLUDED_WORKERS and ann["fine_score"] <= 100
    ]


def coarse_gold(annotations: list[dict]):
    labels = [" ".join(str(a.get("coarse_label", "")).strip().lower().split()) for a in annotations]
    labels = [label for label in labels if label]
    if not labels:
        return None

    def vote(n_positive: int, tie_to_positive: bool) -> bool:
        n_negative = len(labels) - n_positive
        if n_positive > n_negative:
            return True
        if n_positive < n_negative:
            return False
        return tie_to_positive

    n_ok = sum(label == "ok" for label in labels)
    n_missing = sum(label in MISSING_LABELS for label in labels)
    n_added = sum(label in ADDED_LABELS for label in labels)
    return (
        vote(n_ok, True),
        vote(n_missing, False),
        vote(n_added, False),
    )


def coarse_prediction(row: dict) -> tuple[bool, bool, bool]:
    counts = row["error_counts"]
    missing = counts.get("missing", 0)
    extra = counts.get("extra", 0)
    incorrect = counts.get("incorrect", 0)
    return (
        missing == 0 and extra == 0 and incorrect == 0,
        missing > 0 or incorrect > 0,
        extra > 0,
    )


def f1_from_counts(counts: np.ndarray) -> np.ndarray:
    """Return F1 from [..., (TP, FP, FN)] counts."""
    tp, fp, fn = counts[..., 0], counts[..., 1], counts[..., 2]
    denominator = 2.0 * tp + fp + fn
    return np.divide(2.0 * tp, denominator, out=np.zeros_like(denominator, dtype=float), where=denominator > 0)


def safe_corr(function: Callable, x, y) -> float:
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    if len(x_array) < 2:
        return float("nan")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            value = function(x_array, y_array)[0]
    except Exception:
        return float("nan")
    return float(value) if value is not None and np.isfinite(value) else float("nan")


def rowwise_pearson(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pearson correlation for each row of two equally shaped matrices."""
    x_centered = x - np.mean(x, axis=1, keepdims=True)
    y_centered = y - np.mean(y, axis=1, keepdims=True)
    numerator = np.sum(x_centered * y_centered, axis=1)
    denominator = np.sqrt(
        np.sum(x_centered * x_centered, axis=1)
        * np.sum(y_centered * y_centered, axis=1)
    )
    return np.divide(
        numerator,
        denominator,
        out=np.full(numerator.shape, np.nan, dtype=float),
        where=denominator > 0,
    )


def rowwise_kendall_tau_b(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Kendall tau-b for each row, including tie correction."""
    left, right = np.triu_indices(x.shape[1], k=1)
    sign_x = np.sign(x[:, left] - x[:, right])
    sign_y = np.sign(y[:, left] - y[:, right])
    numerator = np.sum(sign_x * sign_y, axis=1)
    denominator = np.sqrt(
        np.sum(sign_x != 0, axis=1) * np.sum(sign_y != 0, axis=1)
    )
    return np.divide(
        numerator,
        denominator,
        out=np.full(numerator.shape, np.nan, dtype=float),
        where=denominator > 0,
    )


def rowwise_correlations(x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "pearson": rowwise_pearson(x, y),
        "spearman": rowwise_pearson(
            scipy.stats.rankdata(x, axis=1),
            scipy.stats.rankdata(y, axis=1),
        ),
        "kendall": rowwise_kendall_tau_b(x, y),
    }


def percentile_summary(values: np.ndarray) -> dict:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        raise ValueError("No finite bootstrap replicates")
    low, high = np.percentile(finite, [2.5, 97.5])
    return {
        "ci_low": float(low),
        "ci_high": float(high),
        "bootstrap_mean": float(np.mean(finite)),
        "n_bootstrap_valid": int(finite.size),
        "ci_excludes_zero": bool(low > 0 or high < 0),
    }


def bootstrap_coarse(
    joint_rows: dict,
    cross_rows: dict,
    annotations: dict,
    n_bootstrap: int,
    seed: int,
    chunk_size: int,
):
    valid_keys = [key for key in joint_rows if coarse_gold(annotations[key]) is not None]
    mr_ids = sorted({key[0] for key in valid_keys}, key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x))
    mr_to_index = {mr_id: index for index, mr_id in enumerate(mr_ids)}
    counts = {
        setting: np.zeros((len(mr_ids), len(TASKS), 3), dtype=np.int64)
        for setting in ("joint", "cross")
    }
    for key in valid_keys:
        gold = coarse_gold(annotations[key])
        cluster = mr_to_index[key[0]]
        for setting, rows in (("joint", joint_rows), ("cross", cross_rows)):
            prediction = coarse_prediction(rows[key])
            for task_index, (truth, pred) in enumerate(zip(gold, prediction)):
                if truth and pred:
                    counts[setting][cluster, task_index, 0] += 1
                elif not truth and pred:
                    counts[setting][cluster, task_index, 1] += 1
                elif truth and not pred:
                    counts[setting][cluster, task_index, 2] += 1

    point = {}
    for setting in ("joint", "cross"):
        task_f1 = f1_from_counts(counts[setting].sum(axis=0))
        point[setting] = {task: float(task_f1[i]) for i, task in enumerate(TASKS)}
        point[setting]["macro"] = float(np.mean(task_f1))

    distributions = {metric: [] for metric in (*TASKS, "macro")}
    rng = np.random.default_rng(seed)
    remaining = n_bootstrap
    while remaining:
        batch = min(chunk_size, remaining)
        sampled = rng.integers(0, len(mr_ids), size=(batch, len(mr_ids)))
        sampled_f1 = {}
        for setting in ("joint", "cross"):
            total_counts = counts[setting][sampled].sum(axis=1)
            values = f1_from_counts(total_counts)
            sampled_f1[setting] = values
        deltas = sampled_f1["cross"] - sampled_f1["joint"]
        for index, task in enumerate(TASKS):
            distributions[task].append(deltas[:, index])
        distributions["macro"].append(np.mean(deltas, axis=1))
        remaining -= batch

    rows = []
    for metric in (*TASKS, "macro"):
        delta_values = np.concatenate(distributions[metric])
        record = {
            "level": "coarse_label",
            "metric": f"{metric}_f1",
            "joint": point["joint"][metric],
            "cross": point["cross"][metric],
            "delta": point["cross"][metric] - point["joint"][metric],
            "n_units": len(mr_ids),
            "n_pairs": len(valid_keys),
            "resampling_unit": "mr_id",
        }
        record.update(percentile_summary(delta_values))
        rows.append(record)
    return rows, {"n_mr_clusters": len(mr_ids), "n_valid_pairs": len(valid_keys)}


def build_correlation_units(rows: dict, annotations: dict):
    by_mr = defaultdict(lambda: {"metric": [], "human": []})
    by_system = defaultdict(lambda: {"metric": [], "human": []})
    for key, row in rows.items():
        anns = annotations[key]
        if not anns:
            continue
        human = float(np.mean([ann["fine_score"] for ann in anns]) / 100.0)
        metric = float(row["f1"])
        if not np.isfinite(metric):
            continue
        mr_id, system = key
        by_mr[mr_id]["metric"].append(metric)
        by_mr[mr_id]["human"].append(human)
        by_system[system]["metric"].append(metric)
        by_system[system]["human"].append(human)
    return by_mr, by_system


def bootstrap_text_correlations(
    joint_rows: dict,
    cross_rows: dict,
    annotations: dict,
    n_bootstrap: int,
    seed: int,
    chunk_size: int,
):
    units = {}
    for setting, rows in (("joint", joint_rows), ("cross", cross_rows)):
        units[setting], _ = build_correlation_units(rows, annotations)
    mr_ids = sorted(
        set(units["joint"]) & set(units["cross"]),
        key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x),
    )
    mr_ids = [
        mr_id for mr_id in mr_ids
        if len(units["joint"][mr_id]["metric"]) >= 2 and len(units["cross"][mr_id]["metric"]) >= 2
    ]

    values = {
        setting: {name: np.empty(len(mr_ids), dtype=float) for name in CORRELATIONS}
        for setting in ("joint", "cross")
    }
    for index, mr_id in enumerate(mr_ids):
        for setting in ("joint", "cross"):
            unit = units[setting][mr_id]
            for name, function in CORRELATIONS.items():
                values[setting][name][index] = safe_corr(function, unit["metric"], unit["human"])

    point = {
        setting: {name: float(np.nanmean(metric_values)) for name, metric_values in setting_values.items()}
        for setting, setting_values in values.items()
    }
    distributions = {name: [] for name in CORRELATIONS}
    rng = np.random.default_rng(seed)
    remaining = n_bootstrap
    while remaining:
        batch = min(chunk_size, remaining)
        sampled = rng.integers(0, len(mr_ids), size=(batch, len(mr_ids)))
        for name in CORRELATIONS:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                joint_value = np.nanmean(values["joint"][name][sampled], axis=1)
                cross_value = np.nanmean(values["cross"][name][sampled], axis=1)
            distributions[name].append(cross_value - joint_value)
        remaining -= batch

    rows = []
    for name in CORRELATIONS:
        record = {
            "level": "text_correlation",
            "metric": f"f1_{name}",
            "joint": point["joint"][name],
            "cross": point["cross"][name],
            "delta": point["cross"][name] - point["joint"][name],
            "n_units": len(mr_ids),
            "n_pairs": sum(bool(annotations[key]) for key in joint_rows),
            "n_joint_finite": int(np.isfinite(values["joint"][name]).sum()),
            "n_cross_finite": int(np.isfinite(values["cross"][name]).sum()),
            "n_paired_finite": int(
                (np.isfinite(values["joint"][name]) & np.isfinite(values["cross"][name])).sum()
            ),
            "resampling_unit": "mr_id",
        }
        record.update(percentile_summary(np.concatenate(distributions[name])))
        rows.append(record)
    return rows, {"n_mr_clusters": len(mr_ids)}


def bootstrap_system_correlations(
    joint_rows: dict,
    cross_rows: dict,
    annotations: dict,
    n_bootstrap: int,
    seed: int,
    chunk_size: int,
):
    system_units = {}
    for setting, rows in (("joint", joint_rows), ("cross", cross_rows)):
        _, system_units[setting] = build_correlation_units(rows, annotations)
    systems = sorted(set(system_units["joint"]) & set(system_units["cross"]))
    human = np.asarray([
        np.mean(system_units["joint"][system]["human"]) for system in systems
    ], dtype=float)
    cross_human = np.asarray([
        np.mean(system_units["cross"][system]["human"]) for system in systems
    ], dtype=float)
    if not np.allclose(human, cross_human, atol=0.0, rtol=0.0):
        raise AssertionError("Human system aggregates differ between paired settings")
    metric = {
        setting: np.asarray([
            np.mean(system_units[setting][system]["metric"]) for system in systems
        ], dtype=float)
        for setting in ("joint", "cross")
    }
    point = {
        setting: {
            name: safe_corr(function, metric[setting], human)
            for name, function in CORRELATIONS.items()
        }
        for setting in ("joint", "cross")
    }

    distributions = {name: [] for name in CORRELATIONS}
    rng = np.random.default_rng(seed)
    remaining = n_bootstrap
    while remaining:
        batch = min(chunk_size, remaining)
        sampled = rng.integers(0, len(systems), size=(batch, len(systems)))
        sampled_human = human[sampled]
        joint_values = rowwise_correlations(metric["joint"][sampled], sampled_human)
        cross_values = rowwise_correlations(metric["cross"][sampled], sampled_human)
        for name in CORRELATIONS:
            distributions[name].append(cross_values[name] - joint_values[name])
        remaining -= batch

    rows = []
    for name in CORRELATIONS:
        record = {
            "level": "system_correlation",
            "metric": f"f1_{name}",
            "joint": point["joint"][name],
            "cross": point["cross"][name],
            "delta": point["cross"][name] - point["joint"][name],
            "n_units": len(systems),
            "n_pairs": len(systems),
            "resampling_unit": "system",
        }
        record.update(percentile_summary(np.concatenate(distributions[name])))
        rows.append(record)
    return rows, {"n_systems": len(systems), "systems": systems}


def git_metadata(repo_root: Path) -> dict:
    def run(*args):
        result = subprocess.run(
            ["git", *args], cwd=repo_root, check=False, capture_output=True, text=True
        )
        return result.returncode, result.stdout.strip(), result.stderr.strip()

    code, commit, _ = run("rev-parse", "HEAD")
    status_code, status, status_error = run("status", "--short", "--branch")
    return {
        "commit": commit if code == 0 else None,
        "dirty": bool(status) if status_code == 0 else None,
        "status_command_error": status_error if status_code != 0 else None,
        "status": status,
    }


def write_outputs(output_dir: Path, records: list[dict], manifest: dict):
    output_dir.mkdir(parents=True, exist_ok=False)
    json_path = output_dir / "results.json"
    csv_path = output_dir / "results.csv"
    manifest_path = output_dir / "manifest.json"
    readme_path = output_dir / "README.md"
    latex_path = output_dir / "table.tex"
    status_path = output_dir / "git_status.txt"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump({"records": records}, handle, indent=2, ensure_ascii=True)
    fieldnames = sorted({key for record in records for key in record})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    status_path.write_text(manifest["git"].pop("status") + "\n", encoding="utf-8")
    manifest["git"]["status_file"] = status_path.name
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=True)

    latex_lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Model & Metric & Joint & Cross & $\Delta$ & 95\% CI \\",
        r"\midrule",
    ]
    for record in records:
        if record["level"] not in {"coarse_label", "text_correlation"}:
            continue
        label = record["metric"].replace("_", r"\_")
        latex_lines.append(
            f'{record["model"]} & {label} & {100*record["joint"]:.1f} & '
            f'{100*record["cross"]:.1f} & {100*record["delta"]:+.1f} & '
            f'[{100*record["ci_low"]:+.1f}, {100*record["ci_high"]:+.1f}] \\\\'
        )
    latex_lines.extend([r"\bottomrule", r"\end{tabular}"])
    latex_path.write_text("\n".join(latex_lines) + "\n", encoding="utf-8")

    key_records = [
        record for record in records
        if record["metric"] in {"is_ok_f1", "has_missing_f1", "has_added_f1", "f1_spearman"}
        and record["level"] in {"coarse_label", "text_correlation"}
    ]
    lines = [
        "# Paired Bootstrap: WebNLG-only to E2E Transfer",
        "",
        "This confirmatory analysis compares WebNLG-only transfer with joint WebNLG+E2E training on identical E2E examples. Delta is `cross-domain - joint-training`. Coarse-label F1 and text-level correlations use paired MR-cluster resampling. System-level correlations are handled separately with paired system resampling.",
        "",
        f"- Bootstrap replicates: {manifest['analysis']['n_bootstrap']:,}",
        f"- Seed: {manifest['analysis']['seed']}",
        f"- Confidence interval: percentile 95%",
        f"- Git commit: `{manifest['git']['commit']}` (dirty: `{manifest['git']['dirty']}`)",
        "",
        "## Key Results",
        "",
        "| Model | Level | Metric | Joint | Cross | Delta | 95% CI | CI excludes 0 |",
        "|---|---|---|---:|---:|---:|---:|:---:|",
    ]
    for record in key_records:
        lines.append(
            f"| {record['model']} | {record['level']} | {record['metric']} | "
            f"{100*record['joint']:.2f} | {100*record['cross']:.2f} | "
            f"{100*record['delta']:+.2f} | "
            f"[{100*record['ci_low']:+.2f}, {100*record['ci_high']:+.2f}] | "
            f"{'yes' if record['ci_excludes_zero'] else 'no'} |"
        )
    lines.extend([
        "",
        "## Interpretation Rule",
        "",
        "A 95% CI containing zero is reported as comparable / not reliably distinguishable under this resampling design. A CI excluding zero supports a directional difference for that metric only.",
        "",
        "## Files",
        "",
        "- `results.json`: complete machine-readable results",
        "- `results.csv`: flat result table",
        "- `table.tex`: paste-ready LaTeX table",
        "- `manifest.json`: estimand, inputs, hashes, software, and command",
        "- `git_status.txt`: complete repository state at execution time",
    ])
    readme_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    started_at = datetime.now(timezone.utc)
    repo_root = args.repo_root.expanduser().resolve()
    converted_path = repo_root / "correlations/e2e19/human_ratings/converted.json"
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else repo_root / "correlations/e2e19/paired_cluster_bootstrap_results"
    )
    output_dir = output_root / started_at.strftime("%Y%m%dT%H%M%SZ")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    if not converted_path.exists():
        raise FileNotFoundError(converted_path)
    converted = load_json(converted_path)

    input_paths = {str(converted_path.relative_to(repo_root)): converted_path}
    records: list[dict] = []
    sample_counts = {}
    for model in args.models:
        paths = {setting: repo_root / relative for setting, relative in MODEL_PAIRS[model].items()}
        for path in paths.values():
            if not path.exists():
                raise FileNotFoundError(path)
            input_paths[str(path.relative_to(repo_root))] = path
        raw = {setting: load_json(path) for setting, path in paths.items()}
        indexed = {setting: index_unique(rows, paths[setting]) for setting, rows in raw.items()}
        joint_keys = set(indexed["joint"])
        cross_keys = set(indexed["cross"])
        if joint_keys != cross_keys:
            raise AssertionError(
                f"{model}: paired key mismatch; joint-only={len(joint_keys-cross_keys)}, "
                f"cross-only={len(cross_keys-joint_keys)}"
            )
        annotations = {key: filtered_annotations(converted, key) for key in joint_keys}

        coarse_records, coarse_counts = bootstrap_coarse(
            indexed["joint"], indexed["cross"], annotations,
            args.n_bootstrap, args.seed, args.chunk_size,
        )
        text_records, text_counts = bootstrap_text_correlations(
            indexed["joint"], indexed["cross"], annotations,
            args.n_bootstrap, args.seed + 1, args.chunk_size,
        )
        system_records, system_counts = bootstrap_system_correlations(
            indexed["joint"], indexed["cross"], annotations,
            args.n_bootstrap, args.seed + 2, args.chunk_size,
        )
        for record in (*coarse_records, *text_records, *system_records):
            record["model"] = model
            record["delta_definition"] = "cross_domain_minus_joint_training"
            records.append(record)
        sample_counts[model] = {
            "n_paired_outputs": len(joint_keys),
            "coarse": coarse_counts,
            "text": text_counts,
            "system": system_counts,
        }

    finished_at = datetime.now(timezone.utc)
    command = shlex.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])
    manifest = {
        "analysis": {
            "status": "confirmatory_paper_reporting",
            "comparison": "WebNLG-only transfer versus joint WebNLG+E2E training on E2E",
            "estimand": "cross-domain minus joint-training",
            "models": args.models,
            "n_bootstrap": args.n_bootstrap,
            "seed": args.seed,
            "derived_seeds": {
                "coarse_label_mr_cluster": args.seed,
                "text_correlation_mr_cluster": args.seed + 1,
                "system_correlation_system_unit": args.seed + 2,
            },
            "ci": "95% percentile bootstrap",
            "coarse_label_definition": "Table 7 majority vote with lenient tie-breaking",
            "human_filter": "drop excluded workers A+C+D+E+F and individual fine scores >100",
            "text_level_definition": "mean within-MR correlation across systems, matching Appendix Table 19",
            "system_level_definition": "correlation across system aggregates, paired resampling by system",
            "sample_counts": sample_counts,
        },
        "inputs": [
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
            }
            for relative, path in sorted(input_paths.items())
        ],
        "execution": {
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": finished_at.isoformat(),
            "command": command,
            "cwd": os.getcwd(),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
        },
        "git": git_metadata(repo_root),
    }
    write_outputs(output_dir, records, manifest)
    print(f"Wrote reproducible outputs to: {output_dir}")
    for record in records:
        if record["level"] in {"coarse_label", "text_correlation"}:
            marker = "excludes 0" if record["ci_excludes_zero"] else "includes 0"
            print(
                f"{record['model']:10s} {record['level']:18s} {record['metric']:18s} "
                f"delta={100*record['delta']:+6.2f} "
                f"CI=[{100*record['ci_low']:+6.2f}, {100*record['ci_high']:+6.2f}] {marker}"
            )


if __name__ == "__main__":
    main()
