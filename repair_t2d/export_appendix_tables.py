#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from table_export_utils import (
    APPENDIX_MODELS,
    BASELINE_LABELS,
    DATASET_LABELS,
    EXPORT_ROOT,
    MODEL_LABELS,
    UPDATED_SUMMARY,
    bold_column_maxima,
    load_all_summaries,
    markdown_table,
    metric_percent,
    source_block,
    write_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export appendix T2D repair tables as Markdown.")
    parser.add_argument("--summary", type=Path, default=UPDATED_SUMMARY)
    parser.add_argument(
        "--output",
        type=Path,
        default=EXPORT_ROOT / "appendix_tables.md",
        help="Markdown output path.",
    )
    return parser.parse_args()


def build_metric_table(dataset: str, baseline: str, metric: str, summaries: dict) -> str:
    headers = ["Method", "Round", *[MODEL_LABELS[model] for model in APPENDIX_MODELS]]
    rows = []

    baseline_row = [f"{BASELINE_LABELS[baseline]} Baseline", "baseline"]
    for model in APPENDIX_MODELS:
        baseline_row.append(metric_percent(summaries["baseline_r1"], dataset, baseline, metric, external=True))
    rows.append(baseline_row)

    row_specs = [
        ("XQDT", "round1", "xqdt_r1", "g_bridge_recover_guard"),
        ("XQDT", "round2", "xqdt_r2", "g_bridge_recover_guard"),
        ("XQDT", "round3", "xqdt_r3", "g_bridge_recover_guard"),
        ("Self Refine", "round1", "baseline_r1", "self_refine_v2"),
        ("Self Refine", "round2", "self_refine_r2", "self_refine_v2"),
        ("Self Refine", "round3", "self_refine_r3", "self_refine_v2"),
        ("FactSpotter", "round1", "baseline_r1", "factspotter_binary"),
        ("FactSpotter", "round2", "baseline_r2", "factspotter_binary"),
        ("FactSpotter", "round3", "baseline_r3", "factspotter_binary"),
        ("NLI", "round1", "baseline_r1", "nli_binary"),
        ("NLI", "round2", "baseline_r2", "nli_binary"),
        ("NLI", "round3", "baseline_r3", "nli_binary"),
        ("PiVe", "round1", "baseline_r1", "pive_missing"),
        ("PiVe", "round2", "baseline_r2", "pive_missing"),
        ("PiVe", "round3", "baseline_r3", "pive_missing"),
    ]

    for method_label, round_label, summary_key, prompt_variant in row_specs:
        row = [method_label, round_label]
        for model in APPENDIX_MODELS:
            summary = summaries[summary_key]
            row.append(
                metric_percent(
                    summary,
                    dataset,
                    baseline,
                    metric,
                    repair_model=model,
                    prompt_variant=prompt_variant,
                )
            )
        rows.append(row)

    section_title = f"### {DATASET_LABELS[dataset].upper()} / {BASELINE_LABELS[baseline]} / {metric} F1 (%)"
    return "\n".join([section_title, "", markdown_table(headers, bold_column_maxima(rows)), ""])


def main() -> None:
    args = parse_args()
    summaries = load_all_summaries(args.summary)
    sections = [
        "# Appendix T2D Repair Tables",
        "",
        "Metrics shown: Exact F1, Strict F1, and Partial F1, all in percentage points.",
        "",
        source_block(
            [
                "baseline_r1",
                "baseline_r2",
                "baseline_r3",
                "self_refine_r2",
                "self_refine_r3",
                "xqdt_r1",
                "xqdt_r2",
                "xqdt_r3",
            ], args.summary
        ),
        "",
    ]

    for dataset in ("webnlg", "genwiki"):
        for baseline in ("bt5", "regen"):
            for metric in ("Exact", "Strict", "Partial"):
                sections.append(build_metric_table(dataset, baseline, metric, summaries))

    write_markdown(args.output, "\n".join(sections))
    print(f"[DONE] wrote markdown: {args.output}")


if __name__ == "__main__":
    main()
