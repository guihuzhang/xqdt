#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Score vLLM simple baseline JSONL outputs with canonical D2T metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from evaluate_repair_text_metrics import compute_metrics, parse_metrics
from score_improve_rounds_full import (
    DEFAULT_METEOR_JAR,
    DEFAULT_METRICS_DIR,
    extract_reference_entries,
    get_reference_texts,
    triples_to_strings,
)


DEFAULT_METRICS = "bertscore,bleu,chrf++,meteor,parent,rouge_l,sbert"
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASETS = {
    "ood472": {
        "aliases": {"ood", "ood472"},
        "tag": "ood",
        "reference_json": PROJECT_ROOT / "inputs" / "ood472_webnlg_en.json",
    },
    "webnlg20": {
        "aliases": {"webnlg", "webnlg20", "webnlgtest"},
        "tag": "webnlgtest",
        "reference_json": PROJECT_ROOT / "inputs" / "webnlg20_control_prefix_20_eval_input.json",
    },
    "genwiki": {
        "aliases": {"genwiki"},
        "tag": "genwiki",
        "reference_json": PROJECT_ROOT / "inputs" / "genwiki_webnlg_en.json",
    },
}


def ensure_file(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")
    return path


def load_json(path: Path) -> Any:
    return json.loads(ensure_file(path).read_text(encoding="utf-8"))


def normalize_dataset_names(names: Optional[Sequence[str]]) -> Set[str]:
    if not names:
        return set(DATASETS)
    selected: Set[str] = set()
    for name in names:
        lowered = name.strip().lower()
        matched = [key for key, spec in DATASETS.items() if lowered in spec["aliases"]]
        if not matched:
            raise ValueError(f"Unknown dataset: {name}")
        selected.update(matched)
    return selected


def parse_prediction_text(prediction: Any) -> str:
    if isinstance(prediction, dict):
        return str(
            prediction.get("full-text")
            or prediction.get("full_text")
            or prediction.get("text")
            or ""
        ).replace("\n", " ").strip()
    if prediction is None:
        return ""
    text = str(prediction).strip()
    try:
        parsed = json.loads(text)
    except Exception:
        return text.replace("\n", " ").strip()
    return parse_prediction_text(parsed)


def read_jsonl_predictions(path: Path) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, int]]:
    by_index: Dict[int, Dict[str, Any]] = {}
    seen_keys: Set[str] = set()
    total_lines = 0
    bad_json = 0
    duplicate_lines = 0

    with ensure_file(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            total_lines += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                bad_json += 1
                continue

            key = str(row.get("key") or row.get("index") or "")
            if key in seen_keys:
                duplicate_lines += 1
            seen_keys.add(key)

            try:
                index = int(row["index"])
            except Exception:
                bad_json += 1
                continue
            by_index[index] = row

    return by_index, {
        "total_lines": total_lines,
        "unique_samples": len(by_index),
        "duplicate_lines": duplicate_lines,
        "bad_json": bad_json,
    }


def collect_metric_inputs(
    reference_entries: Sequence[Dict[str, Any]],
    predictions_by_index: Dict[int, Dict[str, Any]],
    *,
    lang_code: str,
    allow_incomplete: bool,
) -> Tuple[List[List[str]], List[str], List[List[str]]]:
    if not allow_incomplete and len(predictions_by_index) < len(reference_entries):
        raise ValueError(
            f"Incomplete predictions: {len(predictions_by_index)}/{len(reference_entries)}"
        )

    refs: List[List[str]] = []
    hyps: List[str] = []
    graphs: List[List[str]] = []
    for index, ref_entry in enumerate(reference_entries):
        row = predictions_by_index.get(index)
        if row is None:
            if allow_incomplete:
                continue
            raise ValueError(f"Missing prediction index {index}")
        ref_texts = get_reference_texts(ref_entry, lang_code)
        if not ref_texts:
            continue
        refs.append(ref_texts)
        hyps.append(parse_prediction_text(row.get("prediction")))
        graphs.append(triples_to_strings(ref_entry))
    if not refs:
        raise ValueError("No reference texts found for scoring.")
    return refs, hyps, graphs


def output_files_for_dataset(
    spec: Dict[str, Any], model_filter: str, filename_prefix: str, generation_root: Path
) -> List[Path]:
    pattern = f"{filename_prefix}*_{spec['tag']}_desc_none_overlap_dir_none_False_en.jsonl"
    files = sorted(generation_root.rglob(pattern))
    if model_filter:
        files = [path for path in files if model_filter in path.stem]
    return files


def existing_rows(output_path: Path, metrics: Set[str]) -> Dict[str, Dict[str, Any]]:
    if not output_path.is_file():
        return {}
    try:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if set(payload.get("metrics", [])) != set(sorted(metrics)):
        return {}
    rows = payload.get("rows", [])
    return {
        str(row.get("input_file")): row
        for row in rows
        if isinstance(row, dict) and row.get("input_file")
    }


def score_file(
    *,
    dataset_name: str,
    input_file: Path,
    reference_entries: Sequence[Dict[str, Any]],
    metrics: Set[str],
    meteor_jar: Path,
    language: str,
    metrics_dir: Path,
    meteor_disable_paraphrase: bool,
    lang_code: str,
    allow_incomplete: bool,
) -> Dict[str, Any]:
    predictions_by_index, jsonl_meta = read_jsonl_predictions(input_file)
    refs, hyps, graphs = collect_metric_inputs(
        reference_entries,
        predictions_by_index,
        lang_code=lang_code,
        allow_incomplete=allow_incomplete,
    )
    scores = compute_metrics(
        refs,
        hyps,
        graphs,
        metrics=metrics,
        meteor_jar=meteor_jar,
        language=language,
        metrics_dir=metrics_dir,
        meteor_disable_paraphrase=meteor_disable_paraphrase,
        log_prefix=f"{dataset_name} {input_file.stem} ",
    )
    return {
        "dataset": dataset_name,
        "model": input_file.stem,
        "input_file": str(input_file),
        "num_samples": len(refs),
        "jsonl_meta": jsonl_meta,
        "metrics": scores,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", default=None)
    parser.add_argument("--model_filter", type=str, default="")
    parser.add_argument("--filename_prefix", type=str, default="vllm_simple_")
    parser.add_argument(
        "--generation_root",
        type=Path,
        default=PROJECT_ROOT,
    )
    parser.add_argument("--metrics", type=str, default=DEFAULT_METRICS)
    parser.add_argument("--lang_code", type=str, default="en")
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--metrics_dir", type=Path, default=DEFAULT_METRICS_DIR)
    parser.add_argument("--meteor_jar", type=Path, default=DEFAULT_METEOR_JAR)
    parser.add_argument("--meteor_disable_paraphrase", action="store_true")
    parser.add_argument("--allow_incomplete", action="store_true")
    parser.add_argument("--no_resume_scores", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "baseline_outputs" / "vllm_simple_full_metrics_summary.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = parse_metrics(args.metrics, include_meteor=False)
    dataset_names = normalize_dataset_names(args.dataset)
    output_path = args.output.expanduser().resolve()
    generation_root = args.generation_root.expanduser().resolve()
    metrics_dir = args.metrics_dir.expanduser().resolve()
    meteor_jar = args.meteor_jar.expanduser().resolve()
    cached = {} if args.no_resume_scores else existing_rows(output_path, metrics)

    rows: List[Dict[str, Any]] = []
    for dataset_name in sorted(DATASETS):
        spec = DATASETS[dataset_name]
        files = output_files_for_dataset(
            spec, args.model_filter, args.filename_prefix, generation_root
        )
        if dataset_name in dataset_names and not files:
            print(
                f"[skip] no simple-output jsonl files for {dataset_name} under {generation_root}; "
                "pass --generation_root explicitly if they live elsewhere.",
                flush=True,
            )
        if dataset_name not in dataset_names:
            rows.extend(
                row for row in cached.values() if row.get("dataset") == dataset_name
            )
            continue

        reference_entries = extract_reference_entries(
            load_json(Path(spec["reference_json"])),
            Path(spec["reference_json"]),
        )
        for input_file in files:
            key = str(input_file.resolve())
            cached_row = cached.get(key)
            if cached_row and cached_row.get("jsonl_meta", {}).get("unique_samples") == len(reference_entries):
                print(f"[SKIP score] {dataset_name} {input_file.name}")
                rows.append(cached_row)
                continue
            try:
                print(f"[SCORE] {dataset_name} {input_file.name}", flush=True)
                rows.append(
                    score_file(
                        dataset_name=dataset_name,
                        input_file=input_file,
                        reference_entries=reference_entries,
                        metrics=metrics,
                        meteor_jar=meteor_jar,
                        language=args.language,
                        metrics_dir=metrics_dir,
                        meteor_disable_paraphrase=args.meteor_disable_paraphrase,
                        lang_code=args.lang_code,
                        allow_incomplete=args.allow_incomplete,
                    )
                )
            except Exception as exc:
                print(f"[SKIP score] {dataset_name} {input_file.name}: {exc}", flush=True)

    rows.sort(key=lambda row: (row.get("dataset", ""), row.get("model", "")))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metrics": sorted(metrics),
        "score_scale": "0-100 for bounded metrics",
        "rows": rows,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] wrote {output_path}")


if __name__ == "__main__":
    main()
