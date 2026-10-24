#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge multiple eval_summary_text_to_data_repair.json files into one top-level summary.")
    parser.add_argument("--summary-files", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluator", type=str, default=None)
    return parser.parse_args()


def load_summary(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing summary file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Malformed summary payload: {path}")
    return payload


def merge_runs(summary_files: List[Path], evaluator_override: str | None) -> Dict[str, Any]:
    merged_runs: List[Dict[str, Any]] = []
    reference_xmls: Dict[str, str] = {}
    evaluator: str | None = evaluator_override

    for path in summary_files:
        payload = load_summary(path)
        if evaluator is None:
            evaluator = str(payload.get("evaluator", ""))
        for dataset, ref_path in payload.get("reference_xmls", {}).items():
            reference_xmls.setdefault(str(dataset), str(ref_path))
        runs = payload.get("runs", [])
        if not isinstance(runs, list):
            raise ValueError(f"Malformed runs list in {path}")
        merged_runs.extend(runs)

    grouped: Dict[str, Dict[str, Any]] = {}
    for result in merged_runs:
        dataset = str(result["dataset"])
        baseline = str(result["baseline"])
        dataset_group = grouped.setdefault(dataset, {})
        baseline_group = dataset_group.setdefault(baseline, {"external": None, "repairs": {}})
        if result.get("kind") == "external":
            baseline_group["external"] = result
        else:
            repair_model = str(result["repair_model"])
            prompt_variant = str(result["prompt_variant"])
            repairs_group = baseline_group["repairs"].setdefault(repair_model, {})
            repairs_group[prompt_variant] = result

    return {
        "evaluator": evaluator or "",
        "reference_xmls": reference_xmls,
        "runs": merged_runs,
        "grouped": grouped,
    }


def main() -> None:
    args = parse_args()
    merged = merge_runs(args.summary_files, args.evaluator)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] wrote merged summary: {args.output}")


if __name__ == "__main__":
    main()
