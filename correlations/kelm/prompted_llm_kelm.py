#!/usr/bin/env python3
"""Run a prompted LLM on the 60 KELM samples and prepare human verification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from tqdm import tqdm

COMMON_DIR = Path(__file__).resolve().parents[1]
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from prompted_llm_common import (
    add_backend_args,
    append_jsonl,
    build_prompt,
    load_jsonl,
    make_backend,
    model_slug,
    parse_response,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "selected_kelm_10per_size_diverse.json"
DEFAULT_OUTPUT = SCRIPT_DIR / "prompted_llm_results"


def lexicalisation(entry):
    values = entry.get("lexicalisations", {}).get("en", [])
    if not values:
        raise ValueError(f"No English lexicalisation for {entry.get('xml_id')}")
    value = values[0]
    return value.get("lex", "") if isinstance(value, dict) else str(value)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_backend_args(parser)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    data = payload.get("entries", payload)
    if args.max_samples:
        data = data[: args.max_samples]
    slug = model_slug(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / f"{slug}_predictions.jsonl"
    final_path = args.output_dir / f"{slug}_annotation_data.jsonl"
    if args.overwrite:
        checkpoint.unlink(missing_ok=True)
        final_path.unlink(missing_ok=True)
    existing = load_jsonl(checkpoint)

    pending = []
    for index, entry in enumerate(data):
        sample_id = str(entry.get("xml_id", index))
        if sample_id not in existing:
            triples = entry.get("modifiedtripleset", [])
            text = lexicalisation(entry)
            pending.append((sample_id, entry, text, triples, build_prompt(text, triples)))

    backend = make_backend(args)
    try:
        outputs = backend.infer([item[4] for item in pending])
        for (sample_id, entry, text, triples, prompt), output in tqdm(
            zip(pending, outputs), total=len(pending), desc="Saving", unit="sample"
        ):
            parsed = parse_response(output)
            record = {
                "sample_id": sample_id,
                "model": args.model,
                "size": str(entry.get("size", len(triples))),
                "triples": triples,
                "reference_text": text,
                "prompt": prompt,
                "model_response": output,
                "parsed_errors": parsed,
            }
            append_jsonl(checkpoint, record)
            existing[sample_id] = record
    finally:
        backend.close()

    with final_path.open("w", encoding="utf-8") as stream:
        for index, entry in enumerate(data):
            sample_id = str(entry.get("xml_id", index))
            record = existing[sample_id]
            annotation_record = {
                "instance_id": f"{slug}_{sample_id}",
                "size": record["size"],
                "triples": record["triples"],
                "num_triples": len(record["triples"]),
                "reference_text": record["reference_text"],
                "model_response": record["model_response"],
                "parsed_errors": record["parsed_errors"],
                "error_counts": {key: len(value) for key, value in record["parsed_errors"].items()},
                "human_annotation": None,
                "notes": "",
                "model_name": slug,
                "requires_human_verification": True,
            }
            stream.write(json.dumps(annotation_record, ensure_ascii=False) + "\n")
    print(f"Saved {len(data)} annotation records to {final_path}")
    print("Human verification is required before comparing this model with Table 8.")


if __name__ == "__main__":
    main()
