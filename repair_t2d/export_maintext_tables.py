#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from table_export_utils import (
    BASELINE_LABELS,
    DATASET_LABELS,
    EXPORT_ROOT,
    MAIN_TEXT_MODELS,
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
    parser = argparse.ArgumentParser(description="Export main-text T2D repair tables as Markdown.")
    parser.add_argument("--summary", type=Path, default=UPDATED_SUMMARY)
    parser.add_argument(
        "--output",
        type=Path,
        default=EXPORT_ROOT / "maintext_tables.md",
        help="Markdown output path.",
    )
    return parser.parse_args()


def build_table(dataset: str, summaries: dict) -> str:
    headers = ["Method", "Round"]
    for baseline in ("bt5", "regen"):
        for model in MAIN_TEXT_MODELS:
            headers.append(f"{BASELINE_LABELS[baseline]} {MODEL_LABELS[model]}")

    rows = []

    baseline_row = ["Extractor Baseline", "baseline"]
    for baseline in ("bt5", "regen"):
        for model in MAIN_TEXT_MODELS:
            baseline_row.append(metric_percent(summaries["baseline_r1"], dataset, baseline, "Exact", external=True))
    rows.append(baseline_row)

    method_specs = [
        ("XQDT", "round1", "xqdt_r1", "g_bridge_recover_guard"),
        ("XQDT", "round3", "xqdt_r3", "g_bridge_recover_guard"),
        ("Self Refine", "round1", "baseline_r1", "self_refine_v2"),
        ("Self Refine", "round3", "self_refine_r3", "self_refine_v2"),
        ("FactSpotter", "round1", "baseline_r1", "factspotter_binary"),
        ("FactSpotter", "round3", "baseline_r3", "factspotter_binary"),
        ("NLI", "round1", "baseline_r1", "nli_binary"),
        ("NLI", "round3", "baseline_r3", "nli_binary"),
        ("PiVe", "round1", "baseline_r1", "pive_missing"),
        ("PiVe", "round3", "baseline_r3", "pive_missing"),
    ]

    for method_label, round_label, summary_key, prompt_variant in method_specs:
        row = [method_label, round_label]
        for baseline in ("bt5", "regen"):
            for model in MAIN_TEXT_MODELS:
                summary = summaries[summary_key]
                row.append(
                    metric_percent(
                        summary,
                        dataset,
                        baseline,
                        "Exact",
                        repair_model=model,
                        prompt_variant=prompt_variant,
                    )
                )
        rows.append(row)

    formatted_rows = bold_column_maxima(rows)
    title = "Table 9" if dataset == "genwiki" else "Table 10"
    return "\n".join(
        [
            f"## {title} / {DATASET_LABELS[dataset]}",
            "",
            "Metric shown: Exact F1 (%).",
            "",
            markdown_table(headers, formatted_rows),
        ]
    )


def main() -> None:
    args = parse_args()
    summaries = load_all_summaries(args.summary)
    sections = [
        "# Main-Text T2D Repair Tables",
        "",
        "These tables are computed from the bundled updated-profile evaluation results.",
        "",
        source_block(["baseline_r1", "baseline_r3", "self_refine_r3", "xqdt_r1", "xqdt_r3"], args.summary),
        "",
        build_table("genwiki", summaries),
        "",
        build_table("webnlg", summaries),
    ]
    write_markdown(args.output, "\n".join(sections))
    print(f"[DONE] wrote markdown: {args.output}")


if __name__ == "__main__":
    main()
