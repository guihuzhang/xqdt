#!/usr/bin/env python3
"""Build de-identified, language-level evaluation files from source annotations."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_DIR = SCRIPT_DIR / "source_annotations"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "by_language"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def normalize_rating(value: str) -> float:
    return (float(value) - 1.0) / 4.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    items = read_csv(source_dir / "items.csv")
    annotators = read_csv(source_dir / "annotators.csv")
    annotations = read_csv(source_dir / "annotations.csv")
    correction_rows = read_csv(source_dir / "annotation_corrections.csv")

    annotation_by_key = {
        (row["item_id"], row["annotator_id"]): row for row in annotations
    }
    corrections = {
        (row["item_id"], row["annotator_id"], row["field"]): row["corrected_value"]
        for row in correction_rows
    }

    selected_by_language: dict[str, list[dict[str, str]]] = defaultdict(list)
    for annotator in annotators:
        if annotator["included_in_evaluation"].lower() == "true":
            selected_by_language[annotator["language"]].append(annotator)
    for selected in selected_by_language.values():
        selected.sort(key=lambda row: int(row["evaluation_order"]))

    items_by_language: dict[str, list[dict[str, str]]] = defaultdict(list)
    for item in items:
        if item["level"] == "2" and item["language"] in selected_by_language:
            items_by_language[item["language"]].append(item)
    for language_items in items_by_language.values():
        language_items.sort(key=lambda row: int(row["question_id"]))

    output_dir.mkdir(parents=True, exist_ok=True)
    for language in sorted(items_by_language):
        selected = selected_by_language[language]
        records = []
        for index, item in enumerate(items_by_language[language]):
            precision_list = []
            recall_list = []
            for annotator in selected:
                annotator_id = annotator["annotator_id"]
                key = (item["item_id"], annotator_id)
                if key not in annotation_by_key:
                    raise ValueError(f"Missing annotation for {key}")
                annotation = annotation_by_key[key]
                for field, values in (
                    ("precision", precision_list),
                    ("recall", recall_list),
                ):
                    value = corrections.get((*key, field), annotation[field])
                    if not value:
                        raise ValueError(f"Missing {field} rating for {key}")
                    values.append(normalize_rating(value))

            records.append(
                {
                    "id": f"{language}_{index}",
                    "question_id": int(item["question_id"]),
                    "level": int(item["level"]),
                    "language": language,
                    "text": item["text"],
                    "graph": item["graph"],
                    "precision_list": precision_list,
                    "recall_list": recall_list,
                    "num_annotators": len(selected),
                    "annotator_ids": [row["annotator_id"] for row in selected],
                }
            )

        output_path = output_dir / f"{language.lower()}.json"
        output_path.write_text(
            json.dumps(records, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {len(records)} records to {output_path}")


if __name__ == "__main__":
    main()
