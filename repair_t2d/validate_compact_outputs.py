#!/usr/bin/env python3
"""Validate compact T2D traces against the historical full per-round outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import reconstruct_compact_outputs as compact
from build_compact_outputs import BASELINES, DATASETS, PAPER_MODELS


HERE = Path(__file__).resolve().parent
METHODS = ("xqdt", "factspotter", "nli", "pive", "self_refine")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, default=HERE / "t2d_repair_input")
    parser.add_argument("--results-dir", type=Path, default=HERE / "results" / "repair_outputs")
    parser.add_argument("--max-mismatches", type=int, default=20)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def entry_triples(entry: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for item in entry.get("triplets", []):
        if not isinstance(item, dict):
            continue
        out.append(
            f'{str(item.get("subject", "")).strip()} | '
            f'{str(item.get("predicate", "")).strip()} | '
            f'{str(item.get("object", "")).strip()}'
        )
    return out


def expected_output_path(
    root: Path,
    method: str,
    round_id: int,
    dataset: str,
    baseline: str,
    model: str,
) -> Path:
    filename = compact.output_filename(method, model, baseline, dataset)
    if method == "xqdt":
        if round_id == 1:
            path = root / "xqdt_output" / "round1" / "markdown" / dataset / filename
        else:
            path = root / "xqdt_output" / f"round{round_id}" / model / "markdown" / dataset / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    if method == "self_refine":
        if round_id == 1:
            path = root / "baseline" / "round1" / "markdown" / dataset / filename
            if not path.is_file():
                raise FileNotFoundError(path)
            return path
        group_root = root / "baseline" / f"round{round_id}_self_refine"
        matches = sorted(group_root.glob(f"*_r{round_id}/markdown/{dataset}/{filename}"))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected one self-refine output for r{round_id}/{dataset}/{baseline}/{model}; "
                f"found {len(matches)}"
            )
        return matches[0]

    if round_id == 1:
        path = root / "baseline" / "round1" / "markdown" / dataset / filename
    else:
        path = root / "baseline" / f"round{round_id}_baseline" / model / "markdown" / dataset / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def validate_one(
    source_root: Path,
    input_root: Path,
    results_dir: Path,
    method: str,
    dataset: str,
    baseline: str,
    model: str,
    mismatches: List[str],
    max_mismatches: int,
) -> int:
    unified_path, baseline_path = compact.input_paths(input_root, dataset, baseline)
    unified = load_json(unified_path)
    baseline_predictions = load_json(baseline_path)
    sample_ids = [str(row["sample_id"]) for row in unified]
    current: Dict[str, List[str]] = {
        sample_id: compact.normalize_baseline_predictions(baseline_predictions[idx])
        for idx, sample_id in enumerate(sample_ids)
    }

    records_path = compact.method_records_path(results_dir, method, dataset)
    records = compact.load_records(records_path, dataset, baseline, model)
    cursor = 0
    checked = 0
    for round_id in (1, 2, 3):
        while cursor < len(records) and int(records[cursor]["round"]) <= round_id:
            row = records[cursor]
            sample_id = str(row["sample_id"])
            if "triples" in row:
                current[sample_id] = [str(item) for item in row["triples"]]
            elif not bool(row.get("fell_back", False)):
                raise ValueError(
                    f"Invalid compact record without triples/fell_back: {method}/{dataset}/{baseline}/{model}/r{round_id}/{sample_id}"
                )
            cursor += 1

        expected_path = expected_output_path(
            source_root, method, round_id, dataset, baseline, model
        )
        expected_payload = load_json(expected_path)
        expected = {
            str(entry["sample_id"]): entry_triples(entry)
            for entry in expected_payload.get("entries", [])
        }
        if set(expected) != set(sample_ids):
            mismatches.append(
                f"sample-id-set mismatch {method}/{dataset}/{baseline}/{model}/r{round_id}: "
                f"expected={len(expected)} shared={len(sample_ids)}"
            )
            if len(mismatches) >= max_mismatches:
                return checked
        for sample_id in sample_ids:
            checked += 1
            if expected.get(sample_id) != current[sample_id]:
                mismatches.append(
                    f"{method}/{dataset}/{baseline}/{model}/r{round_id}/{sample_id}: "
                    f"expected={expected.get(sample_id)!r} reconstructed={current[sample_id]!r}"
                )
                if len(mismatches) >= max_mismatches:
                    return checked
    return checked


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    mismatches: List[str] = []
    checked = 0
    combinations = 0
    for method in METHODS:
        for model in PAPER_MODELS:
            for dataset in DATASETS:
                for baseline in BASELINES:
                    combinations += 1
                    checked += validate_one(
                        source_root,
                        args.input_root,
                        args.results_dir,
                        method,
                        dataset,
                        baseline,
                        model,
                        mismatches,
                        args.max_mismatches,
                    )
                    if mismatches:
                        break
                if mismatches:
                    break
            if mismatches:
                break
        if mismatches:
            break

    if mismatches:
        print(f"[FAIL] combinations_started={combinations} example_round_rows_checked={checked}")
        for message in mismatches:
            print(message)
        raise SystemExit(1)
    print(
        f"[OK] combinations={combinations} round_outputs={combinations * 3} "
        f"example_round_rows_checked={checked} mismatches=0"
    )


if __name__ == "__main__":
    main()
