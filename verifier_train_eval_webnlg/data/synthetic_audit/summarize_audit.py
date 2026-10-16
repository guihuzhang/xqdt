#!/usr/bin/env python3
"""Summarize the released synthetic-data audit decisions."""

import argparse
import csv
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=Path(__file__).resolve().parent / "synthetic_audit_adjudicated.csv")
    args = parser.parse_args()
    with args.input.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 180 or len({row["audit_id"] for row in rows}) != 180:
        raise ValueError("Expected 180 distinct audit examples.")
    if Counter(row["dataset"] for row in rows) != {"webnlg_train": 90, "e2e_train": 90}:
        raise ValueError("Expected 90 examples from each dataset.")
    if Counter(row["target_error_category"] for row in rows) != {"Extra": 60, "Omitted": 60, "Incorrect": 60}:
        raise ValueError("Expected 60 examples for each error type.")
    fields = ["label_validity_yes_no", "label_completeness_yes_no"]
    for row in rows:
        for field in fields:
            if row[field] not in {"Yes", "No"}:
                raise ValueError(f"{row['audit_id']}: missing or invalid {field}")
    print("| Error type | Perturbation validity | Label completeness |")
    print("| --- | ---: | ---: |")
    for category in ["Extra", "Omitted", "Incorrect", "Overall"]:
        selected = rows if category == "Overall" else [
            row for row in rows if row["target_error_category"] == category
        ]
        cells = []
        for field in fields:
            count = sum(row[field] == "Yes" for row in selected)
            cell = f"{count}/{len(selected)}"
            if category == "Overall":
                cell += f" ({100 * count / len(selected):.1f}%)"
            cells.append(cell)
        print("| " + " | ".join([category, *cells]) + " |")


if __name__ == "__main__":
    main()
