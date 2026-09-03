#!/usr/bin/env python3
"""Evaluate prompted LLMs against English 4L-RP-Human judgements."""

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
DEFAULT_INPUT = SCRIPT_DIR / "by_language/english.json"
DEFAULT_OUTPUT = SCRIPT_DIR / "prompted_llm_results"


def compute_prf(errors, num_triples):
    missing = len(errors["missing"])
    extra = len(errors["extra"])
    incorrect = len(errors["incorrect"])
    tp = max(0, num_triples - missing - incorrect)
    fp = extra + incorrect
    fn = missing + incorrect
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "TP": tp, "FP": fp, "FN": fn}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_backend_args(parser)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    if args.max_samples:
        data = data[: args.max_samples]
    slug = model_slug(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / f"{slug}_results.jsonl"
    final_path = args.output_dir / f"{slug}_results.json"
    if args.overwrite:
        checkpoint.unlink(missing_ok=True)
        final_path.unlink(missing_ok=True)
    existing = load_jsonl(checkpoint)

    pending = []
    for index, sample in enumerate(data):
        sample_id = str(sample.get("id", index))
        if sample_id not in existing:
            triples = [part.strip() for part in sample["graph"].split("<br>") if part.strip()]
            pending.append((sample_id, sample, triples, build_prompt(sample["text"], triples)))

    backend = make_backend(args)
    try:
        outputs = backend.infer([item[3] for item in pending])
        for (sample_id, sample, triples, prompt), output in tqdm(
            zip(pending, outputs), total=len(pending), desc="Saving", unit="sample"
        ):
            parsed = parse_response(output)
            score = compute_prf(parsed, len(triples))
            record = {
                "sample_id": sample_id,
                "model": args.model,
                "text": sample["text"],
                "triples": triples,
                "prompt": prompt,
                "model_output": output,
                "parsed_errors": parsed,
                **score,
                "human_precision_list": sample.get("precision_list", []),
                "human_recall_list": sample.get("recall_list", []),
            }
            append_jsonl(checkpoint, record)
            existing[sample_id] = record
    finally:
        backend.close()

    ordered = [existing[str(sample.get("id", i))] for i, sample in enumerate(data)]
    final_path.write_text(json.dumps(ordered, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(ordered)} results to {final_path}")
    print("Run score_4lrp_from_results.py on the output directory to compute correlations.")


if __name__ == "__main__":
    main()
