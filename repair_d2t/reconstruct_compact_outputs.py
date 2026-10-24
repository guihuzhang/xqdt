#!/usr/bin/env python3
"""Reconstruct per-round D2T texts from the compact released repair records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "inputs" / "webnlg20_control_prefix_20_eval_input.json"
DEFAULT_RESULTS = HERE / "results" / "repair_outputs"
METHOD_FILES = {
    "xqdt": "xqdt.jsonl",
    "factspotter": "factspotter.jsonl",
    "nli": "nli.jsonl",
    "self_refine": "self_refine.jsonl",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=sorted(METHOD_FILES), required=True)
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_records(path: Path, model_tag: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("model_tag") == model_tag:
                records.append(row)
    return sorted(records, key=lambda row: (int(row["round"]), int(row["index"])))


def stage_payload(texts: list[str]) -> dict[str, Any]:
    return {
        "entries": [
            {"lexicalisations": {"en": [{"comment": "", "lex": text}]}}
            for text in texts
        ]
    }


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def reconstruct(
    method: str,
    model_tag: str,
    input_path: Path,
    results_dir: Path,
    output_dir: Path,
) -> None:
    source_rows = load_json(input_path)
    if not isinstance(source_rows, list):
        raise ValueError(f"Expected a JSON list in {input_path}")

    texts = [str(row["input_text"]) for row in source_rows]
    records_path = results_dir / METHOD_FILES[method]
    records = load_records(records_path, model_tag)
    if not records:
        raise ValueError(
            f"No compact records for method={method!r}, model_tag={model_tag!r}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "round0_baseline.json", stage_payload(texts))

    cursor = 0
    for round_id in (1, 2, 3):
        while cursor < len(records) and int(records[cursor]["round"]) <= round_id:
            row = records[cursor]
            index = int(row["index"])
            if index < 0 or index >= len(texts):
                raise IndexError(f"Record index {index} is outside 0..{len(texts)-1}")
            texts[index] = str(row["repaired_text"])
            cursor += 1
        write_json(output_dir / f"round{round_id}_improved.json", stage_payload(texts))

    metadata = {
        "method": method,
        "model_tag": model_tag,
        "num_examples": len(texts),
        "num_sparse_records": len(records),
        "source_input": str(input_path),
        "source_records": str(records_path),
    }
    write_json(output_dir / "reconstruction_summary.json", metadata)


def main() -> None:
    args = parse_args()
    reconstruct(args.method, args.model_tag, args.input, args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
