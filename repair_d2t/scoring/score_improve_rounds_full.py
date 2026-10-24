#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Score improve_vllm_with_eval_loop outputs with canonical full text metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from evaluate_repair_text_metrics import (
    DEFAULT_METEOR_JAR,
    DEFAULT_METRICS_DIR,
    compute_metrics,
    parse_metrics,
)


DEFAULT_METRICS = "bertscore,bleu,chrf++,meteor,parent,rouge_l,sbert"


def ensure_file(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")
    return path


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_text_from_entry(entry: Dict[str, Any], lang_code: str) -> str:
    lexicalisations = entry.get("lexicalisations", {})
    lang_lexes = lexicalisations.get(lang_code) or lexicalisations.get("en") or []
    if lang_lexes:
        first = lang_lexes[0]
        lex = str(first.get("lex", "")).strip()
        if lex:
            return lex
        comment = str(first.get("comment", "")).strip()
        if comment:
            return comment
    return ""


def get_reference_texts(entry: Dict[str, Any], lang_code: str) -> List[str]:
    if isinstance(entry.get("reference_texts"), list):
        refs = [str(ref).replace("\n", " ").strip() for ref in entry["reference_texts"]]
        deduped: List[str] = []
        seen = set()
        for ref in refs:
            if ref and ref not in seen:
                deduped.append(ref)
                seen.add(ref)
        return deduped

    lexicalisations = entry.get("lexicalisations", {})
    lang_lexes = lexicalisations.get(lang_code) or lexicalisations.get("en") or []
    refs: List[str] = []
    for item in lang_lexes:
        lex = str(item.get("lex", "")).strip()
        if lex:
            refs.append(lex.replace("\n", " "))
            continue
        comment = str(item.get("comment", "")).strip()
        if comment:
            refs.append(comment.replace("\n", " "))
    deduped: List[str] = []
    seen = set()
    for ref in refs:
        if ref not in seen:
            deduped.append(ref)
            seen.add(ref)
    return deduped


def normalize_triple_string(triple: Any) -> Optional[str]:
    if isinstance(triple, (list, tuple)) and len(triple) == 3:
        parts = [str(part).strip() for part in triple]
    else:
        parts = [part.strip() for part in str(triple).split(" | ")]
        if len(parts) != 3:
            parts = [part.strip() for part in str(triple).split("|")]
    if len(parts) != 3 or not all(parts):
        return None
    return f"{parts[0]} | {parts[1]} | {parts[2]}"


def triples_to_strings(entry: Dict[str, Any]) -> List[str]:
    if isinstance(entry.get("source_triples"), list):
        triples = [normalize_triple_string(triple) for triple in entry["source_triples"]]
        return [triple for triple in triples if triple]
    if isinstance(entry.get("raw_source_triples"), list):
        triples = [normalize_triple_string(triple) for triple in entry["raw_source_triples"]]
        return [triple for triple in triples if triple]

    triples: List[str] = []
    for triple in entry.get("modifiedtripleset", []):
        normalized = normalize_triple_string([
            triple.get("subject", ""),
            triple.get("property", ""),
            triple.get("object", ""),
        ])
        if normalized:
            triples.append(normalized)
    return triples


def extract_reference_entries(payload: Any, reference_json: Path) -> List[Dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("entries"), list):
        return payload["entries"]
    if isinstance(payload, list):
        return payload
    raise ValueError(
        "Expected reference_json to be either a WebNLG-style dict with an "
        f"'entries' list or an eval_input JSON list: {reference_json}"
    )


def collect_stage_inputs(
    reference_entries: Sequence[Dict[str, Any]],
    stage_entries: Sequence[Dict[str, Any]],
    lang_code: str,
) -> Tuple[List[List[str]], List[str], List[List[str]]]:
    if len(reference_entries) != len(stage_entries):
        raise ValueError(
            f"Reference/stage entry count mismatch: {len(reference_entries)} vs {len(stage_entries)}"
        )

    refs: List[List[str]] = []
    hyps: List[str] = []
    graphs: List[List[str]] = []
    kept = 0
    for ref_entry, stage_entry in zip(reference_entries, stage_entries):
        ref_texts = get_reference_texts(ref_entry, lang_code)
        if not ref_texts:
            continue
        refs.append(ref_texts)
        hyps.append(get_text_from_entry(stage_entry, lang_code).replace("\n", " ").strip())
        graphs.append(triples_to_strings(ref_entry))
        kept += 1
    if kept == 0:
        raise ValueError("No reference texts found in reference_json.")
    return refs, hyps, graphs


def collect_stage_files(run_dir: Path) -> List[Tuple[str, Path]]:
    stages: List[Tuple[str, Path]] = []
    baseline = run_dir / "round0_baseline.json"
    if baseline.is_file():
        stages.append(("round0_baseline", baseline))
    improved_files = sorted(run_dir.glob("round*_improved.json"))
    for path in improved_files:
        if path.stem == "round0_improved":
            continue
        stage_name = path.stem
        stages.append((stage_name, path))
    if not stages:
        raise FileNotFoundError(f"No round baseline/improved JSON files found in {run_dir}")
    return stages


def subtract_scores(current: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Optional[float]]:
    delta: Dict[str, Optional[float]] = {}
    for key in sorted(set(current) | set(baseline)):
        cur = current.get(key)
        base = baseline.get(key)
        if isinstance(cur, (int, float)) and isinstance(base, (int, float)):
            delta[key] = float(cur - base)
        else:
            delta[key] = None
    return delta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--reference_json", type=Path, required=True)
    parser.add_argument("--lang_code", type=str, default="en")
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--metrics", type=str, default=DEFAULT_METRICS)
    parser.add_argument("--include_meteor", action="store_true")
    parser.add_argument("--metrics_dir", type=Path, default=DEFAULT_METRICS_DIR)
    parser.add_argument("--meteor_jar", type=Path, default=DEFAULT_METEOR_JAR)
    parser.add_argument("--meteor_disable_paraphrase", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    reference_json = ensure_file(args.reference_json)
    metrics_dir = args.metrics_dir.expanduser().resolve()
    meteor_jar = args.meteor_jar.expanduser().resolve()
    metrics = parse_metrics(args.metrics, args.include_meteor)

    reference_payload = load_json(reference_json)
    reference_entries = extract_reference_entries(reference_payload, reference_json)

    stages = collect_stage_files(run_dir)
    results: List[Dict[str, Any]] = []
    baseline_scores: Optional[Dict[str, Any]] = None

    for stage_name, stage_path in stages:
        stage_payload = load_json(stage_path)
        if not isinstance(stage_payload, dict) or not isinstance(stage_payload.get("entries"), list):
            raise ValueError(f"Expected stage JSON with an 'entries' list: {stage_path}")
        refs, hyps, graphs = collect_stage_inputs(reference_entries, stage_payload["entries"], args.lang_code)
        scores = compute_metrics(
            refs,
            hyps,
            graphs,
            metrics=metrics,
            meteor_jar=meteor_jar,
            language=args.language,
            metrics_dir=metrics_dir,
            meteor_disable_paraphrase=args.meteor_disable_paraphrase,
            log_prefix=f"{stage_name} ",
        )
        stage_result: Dict[str, Any] = {
            "stage": stage_name,
            "file": str(stage_path),
            "num_samples": len(refs),
            "scores": scores,
        }
        if baseline_scores is None:
            baseline_scores = scores
        else:
            stage_result["delta_vs_round0"] = subtract_scores(scores, baseline_scores)
        results.append(stage_result)

    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else run_dir / "text_metrics" / "round_metrics_full.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_dir": str(run_dir),
        "reference_json": str(reference_json),
        "metrics": sorted(metrics),
        "results": results,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
