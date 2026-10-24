#!/usr/bin/env python3
"""Reconstruct evaluator-compatible T2D R0/R1/R2/R3 outputs from compact traces."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT_ROOT = HERE / "t2d_repair_input"
DEFAULT_RESULTS = HERE / "results" / "repair_outputs"
METHOD_FILES = {
    "xqdt": "xqdt.jsonl",
    "factspotter": "factspotter.jsonl",
    "nli": "nli.jsonl",
    "pive": "pive.jsonl",
    "self_refine": "self_refine.jsonl",
}
PROMPT_VARIANTS = {
    "xqdt": "g_bridge_recover_guard",
    "factspotter": "factspotter_binary",
    "nli": "nli_binary",
    "pive": "pive_missing",
    "self_refine": "self_refine_v2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=sorted(METHOD_FILES), required=True)
    parser.add_argument("--repair-model", required=True)
    parser.add_argument("--dataset", choices=("genwiki", "webnlg"), required=True)
    parser.add_argument("--baseline", choices=("bt5", "regen"), required=True)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def input_paths(root: Path, dataset: str, baseline: str) -> Tuple[Path, Path]:
    unified = root / dataset / f"{dataset}_test_unified.json"
    predictions = root / dataset / "raw" / f"{dataset}_{baseline}.json"
    return unified, predictions


def clean_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def normalize_surface(text: str) -> str:
    value = str(text).replace("_", " ")
    value = re.sub(r"[\"'`“”‘’]", "", value)
    return clean_whitespace(value)


def normalize_predicate(text: str) -> str:
    value = normalize_surface(text).replace("-", " ")
    value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", value)
    return clean_whitespace(value).lower()


def normalize_baseline_predictions(values: Iterable[str]) -> List[str]:
    normalized: List[str] = []
    seen = set()
    for value in values:
        parts = [part.strip() for part in str(value).split("|")]
        if len(parts) != 3:
            continue
        subject, predicate, obj = (
            normalize_surface(parts[0]),
            normalize_predicate(parts[1]),
            normalize_surface(parts[2]),
        )
        if not subject or not predicate or not obj:
            continue
        triple = f"{subject} | {predicate} | {obj}"
        if triple not in seen:
            seen.add(triple)
            normalized.append(triple)
    return normalized


def parse_triple(value: str) -> Dict[str, str]:
    parts = [part.strip() for part in str(value).split("|", 2)]
    if len(parts) != 3 or not all(parts):
        raise ValueError(f"Invalid compact triple: {value!r}")
    return {"subject": parts[0], "predicate": parts[1], "object": parts[2]}


def load_records(
    path: Path,
    dataset: str,
    baseline: str,
    model: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if (
                row.get("dataset") == dataset
                and row.get("baseline") == baseline
                and row.get("model") == model
            ):
                rows.append(row)
    return sorted(rows, key=lambda row: (int(row["round"]), str(row["sample_id"])))


def method_records_path(results_dir: Path, method: str, dataset: str) -> Path:
    if method == "self_refine":
        return results_dir / f"self_refine_{dataset}.jsonl"
    return results_dir / METHOD_FILES[method]


def output_filename(method: str, model: str, baseline: str, dataset: str) -> str:
    variant = PROMPT_VARIANTS[method]
    return f"vllm_text_to_data_{model}_{baseline}_{variant}_{dataset}_0shot_markdown.json"


def baseline_filename(baseline: str, dataset: str) -> str:
    return f"vllm_text_to_data_external_{baseline}_{dataset}_0shot_markdown.json"


def payload(
    sample_ids: List[str],
    current: Dict[str, List[str]],
    *,
    dataset: str,
    baseline: str,
    model: str,
    prompt_variant: str,
) -> Dict[str, Any]:
    return {
        "dataset": dataset,
        "output_style": "markdown",
        "shot_k": 0,
        "prompt_style": prompt_variant,
        "model": model,
        "baseline_name": baseline,
        "entries": [
            {
                "sample_id": sample_id,
                "triplets": [parse_triple(item) for item in current[sample_id]],
            }
            for sample_id in sample_ids
        ],
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    unified_path, baseline_path = input_paths(args.input_root, args.dataset, args.baseline)
    unified = load_json(unified_path)
    baseline_predictions = load_json(baseline_path)
    if not isinstance(unified, list) or not isinstance(baseline_predictions, list):
        raise ValueError("Expected unified input and baseline predictions to be JSON lists")
    if len(unified) != len(baseline_predictions):
        raise ValueError(
            f"Input length mismatch: {len(unified)} unified rows vs {len(baseline_predictions)} baseline predictions"
        )

    sample_ids = [str(row["sample_id"]) for row in unified]
    current: Dict[str, List[str]] = {
        sample_id: normalize_baseline_predictions(baseline_predictions[idx])
        for idx, sample_id in enumerate(sample_ids)
    }
    known_ids = set(sample_ids)

    records_path = method_records_path(args.results_dir, args.method, args.dataset)
    records = load_records(records_path, args.dataset, args.baseline, args.repair_model)
    if not records:
        raise ValueError(
            f"No compact records for method={args.method}, model={args.repair_model}, "
            f"dataset={args.dataset}, baseline={args.baseline}"
        )

    r0_dir = args.output_dir / "round0" / "markdown" / args.dataset
    write_json(
        r0_dir / baseline_filename(args.baseline, args.dataset),
        payload(
            sample_ids,
            current,
            dataset=args.dataset,
            baseline=args.baseline,
            model=f"external_{args.baseline}",
            prompt_variant="external_baseline",
        ),
    )

    cursor = 0
    applied_by_round: Dict[int, int] = {}
    variant = PROMPT_VARIANTS[args.method]
    for round_id in (1, 2, 3):
        applied = 0
        while cursor < len(records) and int(records[cursor]["round"]) <= round_id:
            row = records[cursor]
            sample_id = str(row["sample_id"])
            if sample_id not in known_ids:
                raise KeyError(f"Compact record has unknown sample_id={sample_id}")
            if "triples" in row:
                current[sample_id] = [str(item) for item in row["triples"]]
            elif not bool(row.get("fell_back", False)):
                raise ValueError(f"Compact record for {sample_id} has neither triples nor fell_back=true")
            cursor += 1
            applied += 1
        applied_by_round[round_id] = applied
        round_dir = args.output_dir / f"round{round_id}" / "markdown" / args.dataset
        write_json(
            round_dir / output_filename(
                args.method,
                args.repair_model,
                args.baseline,
                args.dataset,
            ),
            payload(
                sample_ids,
                current,
                dataset=args.dataset,
                baseline=args.baseline,
                model=f"{args.repair_model}_{args.baseline}_{variant}",
                prompt_variant=variant,
            ),
        )

    write_json(
        args.output_dir / "reconstruction_summary.json",
        {
            "method": args.method,
            "repair_model": args.repair_model,
            "dataset": args.dataset,
            "baseline": args.baseline,
            "num_examples": len(sample_ids),
            "num_sparse_records": len(records),
            "records_applied_by_round": applied_by_round,
            "source_input": str(unified_path),
            "source_baseline_predictions": str(baseline_path),
            "source_compact_records": str(records_path),
        },
    )


if __name__ == "__main__":
    main()
