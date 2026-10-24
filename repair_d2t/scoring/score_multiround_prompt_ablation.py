#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Score multi-round improve_vllm_with_eval_loop outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from evaluate_repair_text_metrics import (
    DEFAULT_METEOR_JAR,
    DEFAULT_METRICS_DIR,
    METRIC_OUTPUT_KEYS,
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
    for ref_entry, stage_entry in zip(reference_entries, stage_entries):
        ref_texts = get_reference_texts(ref_entry, lang_code)
        if not ref_texts:
            continue
        refs.append(ref_texts)
        hyps.append(get_text_from_entry(stage_entry, lang_code).replace("\n", " ").strip())
        graphs.append(triples_to_strings(ref_entry))
    if not refs:
        raise ValueError("No reference texts found in reference_json.")
    return refs, hyps, graphs


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


def baseline_cache_key(hyps: Sequence[str]) -> str:
    payload = json.dumps(list(hyps), ensure_ascii=False, sort_keys=False)
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=20).hexdigest()


def read_optional_json(path: Path) -> Optional[Any]:
    if not path.is_file():
        return None
    return load_json(path)


def expected_score_keys(metrics: set[str]) -> set[str]:
    keys: set[str] = set()
    for metric in metrics:
        keys.update(METRIC_OUTPUT_KEYS.get(metric, {metric}))
    return keys


def score_dict_covers(score: Any, expected_keys: set[str]) -> bool:
    return isinstance(score, dict) and all(
        isinstance(score.get(key), (int, float)) and math.isfinite(score[key])
        for key in expected_keys
    )


def existing_result_complete(result: Any, expected_keys: set[str]) -> bool:
    if not isinstance(result, dict):
        return False
    if not score_dict_covers(result.get("before"), expected_keys):
        return False
    scored_rounds = result.get("scored_rounds")
    if not isinstance(scored_rounds, list) or not scored_rounds:
        return False
    for round_idx in scored_rounds:
        if not score_dict_covers(result.get(f"after_round{round_idx}"), expected_keys):
            return False
        if not isinstance(result.get(f"delta_round{round_idx}"), dict):
            return False
    return True


def load_resumable_results(
    output_path: Path,
    reference_json: Path,
    metrics: set[str],
) -> Dict[str, Dict[str, Any]]:
    if not output_path.is_file():
        return {}

    try:
        payload = load_json(output_path)
    except Exception as exc:
        print(f"[resume] failed to load existing metrics {output_path}: {exc}", flush=True)
        return {}

    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        return {}

    existing_ref = str(payload.get("reference_json", ""))
    if existing_ref and Path(existing_ref).name != reference_json.name:
        print(
            f"[resume] ignore existing metrics with different reference_json: {existing_ref}",
            flush=True,
        )
        return {}

    if set(payload.get("metrics", [])) != set(sorted(metrics)):
        print("[resume] ignore existing metrics with different metric set", flush=True)
        return {}

    expected_keys = expected_score_keys(metrics)
    by_run_dir: Dict[str, Dict[str, Any]] = {}
    for result in payload["results"]:
        if not existing_result_complete(result, expected_keys):
            continue
        run_dir_value = result.get("run_dir")
        if not run_dir_value:
            continue
        try:
            key = str(Path(str(run_dir_value)).expanduser().resolve())
        except Exception:
            key = str(run_dir_value)
        by_run_dir[key] = result
    if by_run_dir:
        print(f"[resume] loaded {len(by_run_dir)} completed metric rows from {output_path}", flush=True)
    return by_run_dir


def write_metrics_output(
    output_path: Path,
    reference_json: Path,
    metrics: set[str],
    results: Sequence[Dict[str, Any]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "reference_json": str(reference_json),
        "metrics": sorted(metrics),
        "results": list(results),
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def discover_round_payloads(run_dir: Path) -> Dict[int, Path]:
    rounds: Dict[int, Path] = {}
    round_idx = 1
    while True:
        path = run_dir / f"round{round_idx}_improved.json"
        if not path.is_file():
            break
        rounds[round_idx] = path
        round_idx += 1
    return rounds


def score_run_dir(
    *,
    run_dir: Path,
    reference_entries: Sequence[Dict[str, Any]],
    metrics: set[str],
    meteor_jar: Path,
    language: str,
    metrics_dir: Path,
    meteor_disable_paraphrase: bool,
    lang_code: str,
    baseline_score_cache: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    baseline_path = ensure_file(run_dir / "round0_baseline.json")
    round_payloads = discover_round_payloads(run_dir)
    if not round_payloads:
        raise FileNotFoundError(f"No roundN_improved.json found under {run_dir}")

    baseline_payload = load_json(baseline_path)
    if not isinstance(baseline_payload, dict) or not isinstance(baseline_payload.get("entries"), list):
        raise ValueError(f"Expected an 'entries' list in {baseline_path}")

    refs, baseline_hyps, graphs = collect_stage_inputs(reference_entries, baseline_payload["entries"], lang_code)
    cache_key = baseline_cache_key(baseline_hyps)
    before_cache_hit = cache_key in baseline_score_cache
    if before_cache_hit:
        baseline_scores = dict(baseline_score_cache[cache_key])
        print(f"{run_dir.name} round0 cache hit", flush=True)
    else:
        baseline_scores = compute_metrics(
            refs,
            baseline_hyps,
            graphs,
            metrics=metrics,
            meteor_jar=meteor_jar,
            language=language,
            metrics_dir=metrics_dir,
            meteor_disable_paraphrase=meteor_disable_paraphrase,
            log_prefix=f"{run_dir.name} round0 ",
        )
        baseline_score_cache[cache_key] = dict(baseline_scores)

    result: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "label": run_dir.name,
        "num_samples": len(refs),
        "before_cache_hit": before_cache_hit,
        "before": baseline_scores,
        "scored_rounds": sorted(round_payloads),
        "run_summary": read_optional_json(run_dir / "run_summary.json"),
    }

    for round_idx, round_path in sorted(round_payloads.items()):
        round_payload = load_json(round_path)
        if not isinstance(round_payload, dict) or not isinstance(round_payload.get("entries"), list):
            raise ValueError(f"Expected an 'entries' list in {round_path}")
        _, round_hyps, _ = collect_stage_inputs(reference_entries, round_payload["entries"], lang_code)
        round_scores = compute_metrics(
            refs,
            round_hyps,
            graphs,
            metrics=metrics,
            meteor_jar=meteor_jar,
            language=language,
            metrics_dir=metrics_dir,
            meteor_disable_paraphrase=meteor_disable_paraphrase,
            log_prefix=f"{run_dir.name} round{round_idx} ",
        )
        result[f"round{round_idx}_eval_summary"] = read_optional_json(run_dir / f"round{round_idx}_eval_summary.json")
        result[f"after_round{round_idx}"] = round_scores
        result[f"delta_round{round_idx}"] = subtract_scores(round_scores, baseline_scores)

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, action="append", required=True)
    parser.add_argument("--reference_json", type=Path, required=True)
    parser.add_argument("--lang_code", type=str, default="en")
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--metrics", type=str, default=DEFAULT_METRICS)
    parser.add_argument("--include_meteor", action="store_true")
    parser.add_argument("--metrics_dir", type=Path, default=DEFAULT_METRICS_DIR)
    parser.add_argument(
        "--meteor_jar",
        type=Path,
        default=DEFAULT_METEOR_JAR,
    )
    parser.add_argument("--meteor_disable_paraphrase", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--force_rescore", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference_json = ensure_file(args.reference_json)
    metrics = parse_metrics(args.metrics, args.include_meteor)
    metrics_dir = args.metrics_dir.expanduser().resolve()
    meteor_jar = args.meteor_jar.expanduser().resolve()

    reference_payload = load_json(reference_json)
    reference_entries = extract_reference_entries(reference_payload, reference_json)

    run_dirs = []
    seen_run_dirs = set()
    for path in args.run_dir:
        run_dir = path.expanduser().resolve()
        key = str(run_dir)
        if key in seen_run_dirs:
            print(f"[dedupe] skip duplicate run_dir {run_dir}", flush=True)
            continue
        seen_run_dirs.add(key)
        run_dirs.append(run_dir)

    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else run_dirs[0].parent / "multiround_prompt_ablation_metrics.json"
    )
    resumable_results = (
        {}
        if args.force_rescore
        else load_resumable_results(output_path, reference_json, metrics)
    )
    baseline_score_cache: Dict[str, Dict[str, Any]] = {}
    requested_run_dir_keys = {str(run_dir) for run_dir in run_dirs}
    results_by_run_dir: Dict[str, Dict[str, Any]] = {
        run_dir: result
        for run_dir, result in resumable_results.items()
        if run_dir in requested_run_dir_keys
    }

    def ordered_results() -> List[Dict[str, Any]]:
        return [
            results_by_run_dir[str(run_dir)]
            for run_dir in run_dirs
            if str(run_dir) in results_by_run_dir
        ]

    failed_runs: List[str] = []
    for run_dir in run_dirs:
        run_dir_key = str(run_dir)
        if run_dir_key in results_by_run_dir:
            print(f"{run_dir.name} metrics cache hit", flush=True)
            continue
        try:
            results_by_run_dir[run_dir_key] = score_run_dir(
                run_dir=run_dir,
                reference_entries=reference_entries,
                metrics=metrics,
                meteor_jar=meteor_jar,
                language=args.language,
                metrics_dir=metrics_dir,
                meteor_disable_paraphrase=args.meteor_disable_paraphrase,
                lang_code=args.lang_code,
                baseline_score_cache=baseline_score_cache,
            )
        except Exception as exc:
            failed_runs.append(str(run_dir))
            print(f"[skip] failed to score {run_dir}: {exc}", flush=True)
            continue
        write_metrics_output(output_path, reference_json, metrics, ordered_results())

    write_metrics_output(output_path, reference_json, metrics, ordered_results())
    if failed_runs:
        raise RuntimeError(f"Failed to score {len(failed_runs)} runs: {', '.join(failed_runs)}")
    final_payload = load_json(output_path)
    print(json.dumps(final_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
