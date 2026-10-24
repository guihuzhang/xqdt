#!/usr/bin/env python3
"""Reconstruct saved D2T outputs, recompute metrics, and write Markdown tables."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from reconstruct_compact_outputs import DEFAULT_INPUT, DEFAULT_RESULTS, METHOD_FILES, reconstruct

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "scoring"))

from score_multiround_prompt_ablation import (
    DEFAULT_METEOR_JAR,
    DEFAULT_METRICS,
    DEFAULT_METRICS_DIR,
    parse_metrics,
    score_run_dir,
)

MODEL_LABELS = {
    "gemma3_4b_it": "G3-4B",
    "gemma3_12b_it": "G3-12B",
    "gemma3_27b_it": "G3-27B",
    "llama3_1_8b_instruct": "L3.1-8B",
    "llama3_3_70b_awq": "L3.3-70B",
    "qwen3_4b_nothink": "Q3-4B",
    "qwen3_8b_nothink": "Q3-8B",
    "qwen3_14b_nothink": "Q3-14B",
    "qwen3_32b_nothink": "Q3-32B",
}
METHOD_LABELS = {"self_refine": "SR", "factspotter": "FS", "nli": "NLI", "xqdt": "XQDT"}
METRIC_COLUMNS = [
    ("bleu", "BLEU"),
    ("chrf++", "chrF++"),
    ("meteor", "METEOR"),
    ("parent_f1", "PARENT-F1"),
    ("bertscore_f1", "BERTScore-F1"),
    ("rouge_l", "ROUGE-L"),
    ("sbert_mpnet", "SBERT"),
]


def render_table(results: list[dict[str, Any]], *, main_text: bool) -> str:
    baseline = results[0]["before"]
    candidates = METRIC_COLUMNS[:5] if main_text else METRIC_COLUMNS
    columns = [(key, label) for key, label in candidates if key in baseline]
    headers = ["Model", "Method", "Round"] + [label for _, label in columns]
    separators = ["---"] * 3 + ["---:"] * len(columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(separators) + " |"]

    def append_row(label: str, method: str, round_label: str, scores: dict[str, float]) -> None:
        cells = [label, method, round_label] + [f"{scores[key]:.2f}" for key, _ in columns]
        lines.append("| " + " | ".join(cells) + " |")

    append_row("Control Prefix", "Baseline", "R0", baseline)
    lookup = {(row["model_tag"], row["method"]): row for row in results}
    models = ("gemma3_4b_it", "llama3_1_8b_instruct") if main_text else MODEL_LABELS
    rounds = (1, 3) if main_text else (1, 2, 3)
    for model_tag in models:
        for method, method_label in METHOD_LABELS.items():
            result = lookup.get((model_tag, method))
            if result is None:
                continue
            for round_id in rounds:
                append_row(
                    MODEL_LABELS[model_tag], method_label,
                    f"R{round_id}", result[f"after_round{round_id}"],
                )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--methods", nargs="+", choices=list(METHOD_FILES), default=list(METHOD_FILES),
    )
    parser.add_argument(
        "--model-tags", nargs="+", choices=list(MODEL_LABELS), default=list(MODEL_LABELS),
    )
    parser.add_argument("--metrics", default=DEFAULT_METRICS)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--metrics-dir", type=Path, default=DEFAULT_METRICS_DIR)
    parser.add_argument("--meteor-jar", type=Path, default=DEFAULT_METEOR_JAR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = parse_metrics(args.metrics, False)
    source_rows = json.loads(args.input.read_text(encoding="utf-8"))
    for method in args.methods:
        records_path = args.results_dir / METHOD_FILES[method]
        if not records_path.is_file():
            raise FileNotFoundError(f"Repair records not found: {records_path}")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    results: list[dict[str, Any]] = []
    baseline_cache: dict[str, dict[str, Any]] = {}
    for method in dict.fromkeys(args.methods):
        for model_tag in dict.fromkeys(args.model_tags):
            run_dir = output_dir / "reconstructed" / method / model_tag
            reconstruct(method, model_tag, args.input, args.results_dir, run_dir)
            result = score_run_dir(
                run_dir=run_dir,
                reference_entries=source_rows,
                metrics=metrics,
                meteor_jar=args.meteor_jar,
                language="en",
                metrics_dir=args.metrics_dir,
                meteor_disable_paraphrase=False,
                lang_code="en",
                baseline_score_cache=baseline_cache,
            )
            result.update(
                method=method, model_tag=model_tag,
                run_dir=str(run_dir.relative_to(output_dir)),
            )
            results.append(result)
            payload = {
                "reference_json": args.input.name,
                "num_samples": len(source_rows),
                "metrics": sorted(metrics),
                "results": results,
            }
            (output_dir / "scores.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
            )
    for filename, main_text in (("maintext_tables.md", True), ("appendix_tables.md", False)):
        (output_dir / filename).write_text(render_table(results, main_text=main_text), encoding="utf-8")
    print(f"Scored {len(results)} settings; results and tables: {output_dir}")


if __name__ == "__main__":
    main()
