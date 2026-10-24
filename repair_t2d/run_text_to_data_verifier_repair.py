#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import math
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
INPUT_ROOT = SCRIPT_DIR / "t2d_repair_input"
OUTPUT_ROOT = SCRIPT_DIR / "t2d_repair_output"
DEFAULT_XQDT_OUTPUT_ROOT = OUTPUT_ROOT / "xqdt_output" / "round1"
for import_root in (SCRIPT_DIR, REPO_ROOT):
    import_root_str = str(import_root)
    if import_root_str not in sys.path:
        sys.path.insert(0, import_root_str)

import eval_vllm_ref_less as verifier  # noqa: E402
from generate_vllm_simple import (  # noqa: E402
    build_sampling_params,
    configure_multiprocessing,
    load_vllm,
    render_prompt,
    set_seed,
)
from generate_vllm_text_to_data import (  # noqa: E402
    clean_whitespace,
    deduplicate_triplets,
    format_triplets_markdown,
    normalize_predicate,
    normalize_surface,
    parse_triplets_markdown,
)


UNIFIED_DATASETS = {
    "genwiki": INPUT_ROOT / "genwiki" / "genwiki_test_unified.json",
    "webnlg": INPUT_ROOT / "webnlg" / "webnlg_test_unified.json",
}

PROJECT_BASELINE_PREDICTIONS = {
    "genwiki": {
        "bt5": INPUT_ROOT / "genwiki" / "raw" / "genwiki_bt5.json",
        "regen": INPUT_ROOT / "genwiki" / "raw" / "genwiki_regen.json",
    },
    "webnlg": {
        "bt5": INPUT_ROOT / "webnlg" / "raw" / "webnlg_bt5.json",
        "regen": INPUT_ROOT / "webnlg" / "raw" / "webnlg_regen.json",
    },
}

REPAIR_MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "llama3_3_70b_awq": {
        "model_id": "casperhansen/llama-3.3-70b-instruct-awq",
        "model_type": "llama3.3",
        "model_size": "70b_awq",
        "dtype": "auto",
        "tensor_parallel_size": 1,
        "disable_thinking": False,
    },
    "qwen3_32b": {
        "model_id": "Qwen/Qwen3-32B",
        "model_type": "qwen3",
        "model_size": "32b",
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "disable_thinking": True,
    },
    "gemma3_27b": {
        "model_id": "google/gemma-3-27b-it",
        "model_type": "gemma3",
        "model_size": "27b",
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "disable_thinking": False,
    },
    "qwen3_14b": {
        "model_id": "Qwen/Qwen3-14B",
        "model_type": "qwen3",
        "model_size": "14b",
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "disable_thinking": True,
    },
    "qwen3_8b": {
        "model_id": "Qwen/Qwen3-8B",
        "model_type": "qwen3",
        "model_size": "8b",
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "disable_thinking": True,
    },
    "qwen3_4b": {
        "model_id": "Qwen/Qwen3-4B",
        "model_type": "qwen3",
        "model_size": "4b",
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "disable_thinking": True,
    },
    "qwen3_1.7b": {
        "model_id": "Qwen/Qwen3-1.7B",
        "model_type": "qwen3",
        "model_size": "1.7b",
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "disable_thinking": True,
    },
    "qwen3_0.6b": {
        "model_id": "Qwen/Qwen3-0.6B",
        "model_type": "qwen3",
        "model_size": "0.6b",
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "disable_thinking": True,
    },
    "llama3_1_8b": {
        "model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "model_type": "llama3.1",
        "model_size": "8b",
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "disable_thinking": False,
    },
    "llama3_2_3b": {
        "model_id": "meta-llama/Llama-3.2-3B-Instruct",
        "model_type": "llama3.2",
        "model_size": "3b",
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "disable_thinking": False,
    },
    "llama3_2_1b": {
        "model_id": "meta-llama/Llama-3.2-1B-Instruct",
        "model_type": "llama3.2",
        "model_size": "1b",
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "disable_thinking": False,
    },
    "gemma3_12b": {
        "model_id": "google/gemma-3-12b-it",
        "model_type": "gemma3",
        "model_size": "12b",
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "disable_thinking": False,
    },
    "gemma3_4b": {
        "model_id": "google/gemma-3-4b-it",
        "model_type": "gemma3",
        "model_size": "4b",
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "disable_thinking": False,
    },
    "gemma3_1b": {
        "model_id": "google/gemma-3-1b-it",
        "model_type": "gemma3",
        "model_size": "1b",
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "disable_thinking": False,
    },
}

REPAIR_SYSTEM = "You are a careful text-to-data repair assistant."
VERIFIER_MODEL_KEY = "qwen8b"
VERIFIER_MODEL_NAME = verifier.MODEL_CONFIGS[VERIFIER_MODEL_KEY]["name"]

PROMPT_VARIANTS: Dict[str, Dict[str, Any]] = {
    "g_bridge_recover_guard": {
        "title": "bridge_recover_guard",
        "feedback_notes": {
            "extra": "Review whether each flagged row is truly unsupported by the text. Remove it only when the text does not support that fact, not merely because the wording differs.",
            "incorrect": "If the text makes the intended fact clear, revise the row into a supported form rather than simply deleting it. Prefer a local correction that preserves the row's intended fact.",
            "missing": "Add a triple only when the fact is clearly and explicitly stated in the text, and do not duplicate a fact that is already captured by a supported row.",
        },
        "task_lines": [
            "Use the source text as the final authority.",
            "Treat the review notes as suggestions rather than facts.",
            "When a flagged incorrect row can be repaired into a supported fact, prefer correction over deletion.",
            "When you revise a row, align the subject and object with the wording used in the source text whenever possible.",
            "Do not remove a row only because its wording differs from the text if the same supported fact is still present.",
            "If no unsupported rows are flagged, avoid removing rows unless the text clearly contradicts them.",
            "Keep unaffected triples unchanged.",
            "If you are uncertain, keep the current triple instead of guessing.",
        ],
        "requirement_line": "Prefer supported local correction and source-aligned wording; delete a row only when it is unsupported, contradicted, or clearly replaced by a better supported row.",
    },
}


def ensure_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
        fp.flush()


def save_jsonl_rows(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl_records(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []

    rows: List[Dict[str, Any]] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_idx, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            if line_idx == len(lines):
                print(
                    f"[RESUME] ignoring incomplete trailing jsonl line: {path}:{line_idx}",
                    flush=True,
                )
                break
            raise ValueError(f"Malformed jsonl checkpoint at {path}:{line_idx}") from None
        if not isinstance(payload, dict):
            raise ValueError(f"Malformed jsonl checkpoint row at {path}:{line_idx}")
        rows.append(payload)
    return rows


def parse_tp_overrides(items: Sequence[str]) -> Dict[str, int]:
    overrides: Dict[str, int] = {}
    for item in items:
        key, sep, value = str(item).partition("=")
        key = key.strip()
        value = value.strip()
        if not sep or not key or not value:
            raise ValueError(f"Invalid TP override: {item!r}. Expected <model>=<tp>.")
        if key not in REPAIR_MODEL_CONFIGS:
            raise ValueError(f"Unknown repair model in TP override: {key}")
        try:
            tp = int(value)
        except ValueError as exc:
            raise ValueError(f"Invalid TP size in override {item!r}") from exc
        if tp < 1:
            raise ValueError(f"TP size must be >= 1 in override {item!r}")
        overrides[key] = tp
    return overrides


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run verifier-driven text-to-data repair on project-local or few-shot extraction outputs.")
    parser.add_argument("--datasets", nargs="+", default=["genwiki", "webnlg"], choices=sorted(UNIFIED_DATASETS))
    parser.add_argument("--baselines", nargs="+", default=["bt5", "regen"], choices=["bt5", "regen", "fewshot"])
    parser.add_argument("--repair-models", nargs="+", default=["gemma3_4b", "gemma3_12b", "gemma3_27b", "llama3_1_8b", "llama3_3_70b_awq", "qwen3_4b", "qwen3_8b", "qwen3_14b", "qwen3_32b"], choices=sorted(REPAIR_MODEL_CONFIGS))
    parser.add_argument("--prompt-variants", nargs="+", default=["g_bridge_recover_guard"], choices=sorted(PROMPT_VARIANTS))
    parser.add_argument("--stage", choices=["all", "prepare", "verify", "repair"], default="all")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--input-json", type=Path, default=None, help="Optional previous repair output json to use as the current triples input.")
    parser.add_argument("--only-previously-flagged", action="store_true", help="When used with --input-json, continue only entries whose previous output had flagged=true.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_XQDT_OUTPUT_ROOT)
    parser.add_argument("--model-cache", type=str, default=".cache_text_to_data_repair")
    parser.add_argument("--verifier-temperature", type=float, default=0.3)
    parser.add_argument("--verifier-seed", type=int, default=2023)
    parser.add_argument("--verifier-batch-size", type=int, default=1)
    parser.add_argument("--verifier-max-tokens", type=int, default=1024)
    parser.add_argument("--verifier-dtype", type=str, default="bfloat16")
    parser.add_argument("--verifier-gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--verifier-max-model-len", type=int, default=4096)
    parser.add_argument("--verifier-max-num-seqs", type=int, default=16)
    parser.add_argument("--verifier-max-lora-rank", type=int, default=16)
    parser.add_argument("--repair-temperature", type=float, default=0.7)
    parser.add_argument("--repair-seed", type=int, default=42)
    parser.add_argument("--repair-top-p", type=float, default=0.7)
    parser.add_argument("--repair-top-k", type=int, default=50)
    parser.add_argument("--repair-repetition-penalty", type=float, default=1.0)
    parser.add_argument("--repair-max-new-tokens", type=int, default=1024)
    parser.add_argument("--repair-batch-size", type=int, default=4)
    parser.add_argument("--repair-max-model-len", type=int, default=4096)
    parser.add_argument("--repair-gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--repair-max-num-seqs", type=int, default=None)
    parser.add_argument("--repair-tp-overrides", nargs="*", default=[], help="Optional per-model TP overrides like gemma3_27b=2 qwen3_32b=2")
    parser.add_argument("--shot-k", type=int, default=None, help="Override shot count in output filenames/metadata. Defaults to input payload shot_k or 0.")
    args = parser.parse_args()
    args.repair_tp_override_map = parse_tp_overrides(args.repair_tp_overrides)
    return args


def parse_pipe_triple(triple_text: str) -> Optional[Tuple[str, str, str]]:
    parts = [part.strip() for part in str(triple_text).split("|")]
    if len(parts) != 3:
        return None
    subject, predicate, obj = parts
    normalized = (
        normalize_surface(subject),
        normalize_predicate(predicate),
        normalize_surface(obj),
    )
    if not all(normalized):
        return None
    return normalized


def parse_structured_triple(triple_text: str) -> Optional[Tuple[str, str, str]]:
    triple = clean_whitespace(str(triple_text))
    labeled = re.match(r"^\[S\]\s*(.*?)\s*\[P\]\s*(.*?)\s*\[O\]\s*(.*)$", triple)
    if labeled:
        normalized = (
            normalize_surface(labeled.group(1)),
            normalize_predicate(labeled.group(2)),
            normalize_surface(labeled.group(3)),
        )
        if all(normalized):
            return normalized
    return parse_pipe_triple(triple)


def triples_to_strings(triples: Iterable[Tuple[str, str, str]]) -> List[str]:
    return [f"{subject} | {predicate} | {obj}" for subject, predicate, obj in triples]


def strings_to_tuples(triples: Iterable[str]) -> List[Tuple[str, str, str]]:
    tuples: List[Tuple[str, str, str]] = []
    for triple in triples:
        parsed = parse_structured_triple(triple)
        if parsed is not None:
            tuples.append(parsed)
    return deduplicate_triplets(tuples)


def strings_to_dicts(triples: Iterable[str]) -> List[Dict[str, str]]:
    rows = []
    for subject, predicate, obj in strings_to_tuples(triples):
        rows.append({"subject": subject, "predicate": predicate, "object": obj})
    return rows


def entry_triplets_to_strings(triplets: Iterable[Any]) -> List[str]:
    tuples: List[Tuple[str, str, str]] = []
    for triplet in triplets:
        parsed: Optional[Tuple[str, str, str]] = None
        if isinstance(triplet, dict):
            parsed = (
                normalize_surface(str(triplet.get("subject", ""))),
                normalize_predicate(str(triplet.get("predicate", ""))),
                normalize_surface(str(triplet.get("object", ""))),
            )
            if not all(parsed):
                parsed = None
        elif isinstance(triplet, str):
            parsed = parse_structured_triple(triplet)
        if parsed is not None:
            tuples.append(parsed)
    return triples_to_strings(deduplicate_triplets(tuples))


def markdown_from_strings(triples: Iterable[str]) -> str:
    tuples = strings_to_tuples(triples)
    if not tuples:
        return "| subject | predicate | object |\n|---|---|---|"
    return format_triplets_markdown(tuples)


def load_baseline_samples(
    dataset: str,
    baseline: str,
    max_samples: Optional[int],
    *,
    input_json: Optional[Path] = None,
    only_previously_flagged: bool = False,
) -> List[Dict[str, Any]]:
    unified_path = ensure_file(UNIFIED_DATASETS[dataset], f"{dataset} unified test data")
    unified_samples = load_json(unified_path)
    if not isinstance(unified_samples, list):
        raise ValueError(f"Unexpected unified payload type for {dataset}")

    entry_by_id: Optional[Dict[str, Dict[str, Any]]] = None
    predicted_lists: Optional[List[Any]] = None
    shot_k = 0
    if input_json is not None:
        input_payload = load_json(ensure_file(input_json, f"{dataset}/{baseline} iterative input json"))
        if not isinstance(input_payload, dict):
            raise ValueError(f"Unexpected iterative input payload type for {input_json}")
        payload_dataset = input_payload.get("dataset")
        payload_baseline = input_payload.get("baseline_name")
        payload_shot_k = input_payload.get("shot_k")
        if payload_dataset not in {None, dataset}:
            raise ValueError(f"Iterative input dataset mismatch: expected {dataset}, found {payload_dataset}")
        if payload_baseline not in {None, baseline}:
            raise ValueError(f"Iterative input baseline mismatch: expected {baseline}, found {payload_baseline}")
        if payload_shot_k is not None:
            shot_k = int(payload_shot_k)
        entries = input_payload.get("entries")
        if not isinstance(entries, list):
            raise ValueError(f"Iterative input payload missing entries list: {input_json}")
        entry_by_id = {}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            sample_id = str(entry.get("sample_id", ""))
            if sample_id:
                entry_by_id[sample_id] = entry
    else:
        if baseline == "fewshot":
            raise ValueError("--baselines fewshot requires --input-json.")
        prediction_path = ensure_file(PROJECT_BASELINE_PREDICTIONS[dataset][baseline], f"{dataset}/{baseline} project-local baseline predictions")
        predicted_lists = load_json(prediction_path)
        if not isinstance(predicted_lists, list):
            raise ValueError(f"Unexpected predictions payload type for {dataset}/{baseline}")
        if len(unified_samples) != len(predicted_lists):
            raise ValueError(
                f"Length mismatch for {dataset}/{baseline}: "
                f"{len(unified_samples)} unified samples vs {len(predicted_lists)} predictions"
            )

    if max_samples is not None:
        unified_samples = unified_samples[:max_samples]
        if predicted_lists is not None:
            predicted_lists = predicted_lists[:max_samples]

    samples: List[Dict[str, Any]] = []
    for idx, sample in enumerate(unified_samples):
        sample_id = str(sample["sample_id"])
        if entry_by_id is not None:
            entry = entry_by_id.get(sample_id)
            if entry is None:
                raise ValueError(f"Iterative input missing sample_id={sample_id} for {dataset}/{baseline}")
            normalized_triples = entry_triplets_to_strings(entry.get("triplets", []))
            previous_flagged = bool(entry.get("flagged", False))
            source_text = str(entry.get("source_text") or sample["source_text"])
        else:
            predicted = predicted_lists[idx]
            normalized_triples = triples_to_strings(strings_to_tuples(predicted))
            previous_flagged = False
            source_text = str(sample["source_text"])
        samples.append(
            {
                "index": idx,
                "sample_id": sample_id,
                "dataset": dataset,
                "baseline_name": baseline,
                "source_text": source_text,
                "gold_triples": [str(item) for item in sample.get("gold_triples", [])],
                "raw_prediction_triples": list(normalized_triples),
                "current_triples": normalized_triples,
                "iteration_active": previous_flagged if only_previously_flagged else True,
                "previous_flagged": previous_flagged,
                "shot_k": shot_k,
            }
        )
    return samples


def baseline_output_filename(dataset: str, baseline: str, shot_k: int = 0) -> str:
    return f"vllm_text_to_data_external_{baseline}_{dataset}_{shot_k}shot_markdown.json"


def repair_output_filename(dataset: str, baseline: str, repair_model: str, prompt_variant: str, shot_k: int = 0) -> str:
    return f"vllm_text_to_data_{repair_model}_{baseline}_{prompt_variant}_{dataset}_{shot_k}shot_markdown.json"


def build_entries_payload(
    *,
    dataset: str,
    model_label: str,
    prompt_variant: Optional[str],
    baseline: str,
    shot_k: int,
    entries: List[Dict[str, Any]],
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = {
        "dataset": dataset,
        "output_style": "markdown",
        "shot_k": shot_k,
        "prompt_style": prompt_variant or "external_baseline",
        "model": model_label,
        "baseline_name": baseline,
        "entries": entries,
    }
    if extra_meta:
        payload.update(extra_meta)
    return payload


def save_baseline_payload(samples: List[Dict[str, Any]], output_root: Path) -> Path:
    dataset = str(samples[0]["dataset"])
    baseline = str(samples[0]["baseline_name"])
    shot_k = int(samples[0].get("shot_k", 0))
    entries = []
    for sample in samples:
        entries.append(
            {
                "sample_id": sample["sample_id"],
                "triplets": strings_to_dicts(sample["current_triples"]),
                "raw_output": "",
                "source_text": sample["source_text"],
                "baseline_name": baseline,
            }
        )
    output_path = ensure_dir(output_root / "markdown" / dataset) / baseline_output_filename(dataset, baseline, shot_k)
    payload = build_entries_payload(
        dataset=dataset,
        model_label=f"external_{baseline}",
        prompt_variant=None,
        baseline=baseline,
        shot_k=shot_k,
        entries=entries,
        extra_meta={"source": "external_text_to_data_baseline"},
    )
    save_json(output_path, payload)
    return output_path


def canonicalize_parsed_output(parsed_output: Dict[str, List[str]]) -> Dict[str, List[str]]:
    # The verifier's raw missing/extra labels are reversed. Convert them once here
    # so downstream repair prompt uses the true semantics.
    extra = triples_to_strings(strings_to_tuples(parsed_output.get("missing", [])))
    missing = triples_to_strings(strings_to_tuples(parsed_output.get("extra", [])))
    incorrect = triples_to_strings(strings_to_tuples(parsed_output.get("incorrect", [])))
    return {"extra": extra, "missing": missing, "incorrect": incorrect}


def is_flagged(canonical_feedback: Dict[str, List[str]]) -> bool:
    return any(canonical_feedback.get(name) for name in ("extra", "missing", "incorrect"))


def make_verifier_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        model_cache=args.model_cache,
        dtype=args.verifier_dtype,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.verifier_gpu_memory_utilization,
        max_model_len=args.verifier_max_model_len,
        max_num_seqs=args.verifier_max_num_seqs,
        quantization="",
        enforce_eager=False,
        max_lora_rank=args.verifier_max_lora_rank,
        use_hf="1",
        max_tokens=args.verifier_max_tokens,
        top_p=None,
        top_k=None,
        repetition_penalty=None,
        seed=args.verifier_seed,
        batch_size=args.verifier_batch_size,
    )


def verifier_scope_root(output_root: Path, args: argparse.Namespace) -> Path:
    if args.input_json is not None and len(args.repair_models) == 1:
        return ensure_dir(output_root / args.repair_models[0])
    return output_root


def run_verifier(
    samples: List[Dict[str, Any]],
    args: argparse.Namespace,
    output_root: Path,
) -> List[Dict[str, Any]]:
    dataset = str(samples[0]["dataset"])
    baseline = str(samples[0]["baseline_name"])
    verifier_dir = ensure_dir(verifier_scope_root(output_root, args) / "verifier" / dataset / baseline / VERIFIER_MODEL_NAME)
    output_path = verifier_dir / "verify_results.json"

    if output_path.exists():
        payload = load_json(output_path)
        if isinstance(payload, list):
            return payload
        raise ValueError(f"Malformed verifier cache: {output_path}")

    active_samples = [sample for sample in samples if sample.get("iteration_active", True)]
    verifier_args = make_verifier_args(args)
    verifier.set_seed(args.verifier_seed)
    engine = None
    template = None
    previous_use_hf = None
    rows_by_id: Dict[str, Dict[str, Any]] = {}

    for sample in samples:
        if sample.get("iteration_active", True):
            continue
        rows_by_id[sample["sample_id"]] = {
            "index": sample["index"],
            "sample_id": sample["sample_id"],
            "dataset": dataset,
            "baseline_name": baseline,
            "source_text": sample["source_text"],
            "current_triples": sample["current_triples"],
            "prompt": "",
            "raw_model_output": "",
            "raw_parsed_output": {"extra": [], "missing": [], "incorrect": []},
            "canonical_feedback": {"extra": [], "missing": [], "incorrect": []},
            "flagged": False,
            "skipped_iteration": True,
        }

    if not active_samples:
        rows = [rows_by_id[sample["sample_id"]] for sample in samples]
        save_json(output_path, rows)
        save_json(
            verifier_dir / "summary.json",
            {
                "dataset": dataset,
                "baseline_name": baseline,
                "verifier_model": VERIFIER_MODEL_NAME,
                "num_samples": len(rows),
                "flagged_samples": 0,
                "unflagged_samples": len(rows),
            },
        )
        return rows

    try:
        engine, template, previous_use_hf = verifier.load_vllm_engine(verifier_args, verifier.MODEL_CONFIGS[VERIFIER_MODEL_KEY])
        infer_request_cls = verifier.import_swift_api()["InferRequest"]
        request_config = verifier.build_request_config(verifier_args, args.verifier_temperature)

        total_batches = math.ceil(len(active_samples) / args.verifier_batch_size) if active_samples else 0
        progress_iter = enumerate(range(0, len(active_samples), args.verifier_batch_size), start=1)
        if tqdm is not None:
            progress_iter = enumerate(
                tqdm(
                    range(0, len(active_samples), args.verifier_batch_size),
                    total=total_batches,
                    desc=f"verify {dataset}/{baseline}",
                    dynamic_ncols=True,
                    leave=True,
                ),
                start=1,
            )
        for batch_idx, start in progress_iter:
            batch = active_samples[start : start + args.verifier_batch_size]
            if tqdm is None and (batch_idx == 1 or batch_idx % 25 == 0 or batch_idx == total_batches):
                print(
                    f"[VERIFY-PROGRESS] dataset={dataset} baseline={baseline} "
                    f"batch={batch_idx}/{total_batches} samples={min(start + len(batch), len(active_samples))}/{len(active_samples)}",
                    flush=True,
                )
            infer_requests = []
            prompts = []
            for sample in batch:
                query = verifier.build_query(sample["source_text"], sample["current_triples"])
                prompts.append(query)
                infer_requests.append(
                    verifier.make_infer_request(
                        infer_request_cls,
                        query,
                        verifier.MODEL_CONFIGS[VERIFIER_MODEL_KEY].get("default_system"),
                        pass_system=template is None,
                    )
                )

            infer_kwargs = {"use_tqdm": False}
            if template is not None:
                infer_kwargs["template"] = template
            responses = engine.infer(infer_requests, request_config, **infer_kwargs)

            for sample, prompt, response in zip(batch, prompts, responses):
                raw_output = response.choices[0].message.content
                raw_parsed = verifier.parse_prediction(raw_output)
                canonical = canonicalize_parsed_output(raw_parsed)
                rows_by_id[sample["sample_id"]] = {
                    "index": sample["index"],
                    "sample_id": sample["sample_id"],
                    "dataset": dataset,
                    "baseline_name": baseline,
                    "source_text": sample["source_text"],
                    "current_triples": sample["current_triples"],
                    "prompt": prompt,
                    "raw_model_output": raw_output,
                    "raw_parsed_output": raw_parsed,
                    "canonical_feedback": canonical,
                    "flagged": is_flagged(canonical),
                    "skipped_iteration": False,
                }
    finally:
        verifier.restore_use_hf(previous_use_hf)
        del engine, template
        gc.collect()
        verifier.clear_cuda_memory()

    rows = [rows_by_id[sample["sample_id"]] for sample in samples]
    save_json(output_path, rows)
    summary = {
        "dataset": dataset,
        "baseline_name": baseline,
        "verifier_model": VERIFIER_MODEL_NAME,
        "num_samples": len(rows),
        "flagged_samples": sum(1 for row in rows if row["flagged"]),
        "unflagged_samples": sum(1 for row in rows if not row["flagged"]),
    }
    save_json(verifier_dir / "summary.json", summary)
    return rows


def format_feedback_section(title: str, triples: Sequence[str], note: str) -> str:
    if not triples:
        return f"{title}\n- None\n"
    return f"{title}\n{markdown_from_strings(triples)}\nNote: {note}\n"


def build_repair_user_prompt(
    *,
    variant: str,
    source_text: str,
    current_triples: Sequence[str],
    canonical_feedback: Dict[str, List[str]],
) -> str:
    config = PROMPT_VARIANTS[variant]
    task_block = "\n".join(f"- {line}" for line in config["task_lines"])
    requirement_line = str(config["requirement_line"])
    feedback_notes = config["feedback_notes"]
    return f"""Repair the extracted triples using the source text and verifier feedback.

# Source text
{source_text}

# Current triples
{markdown_from_strings(current_triples)}

# Verifier feedback
{format_feedback_section(
    "Unsupported triples to remove:",
    canonical_feedback.get("extra", []),
    feedback_notes["extra"],
)}
{format_feedback_section(
    "Potentially incorrect triples to check:",
    canonical_feedback.get("incorrect", []),
    feedback_notes["incorrect"],
)}
{format_feedback_section(
    "Potentially missing facts to consider:",
    canonical_feedback.get("missing", []),
    feedback_notes["missing"],
)}
# Task
{task_block}

# Output requirements
- Return only a markdown table with exactly three columns: subject, predicate, object.
- Keep one row per final triple.
- Keep unaffected triples unchanged.
- {requirement_line}
- Do not explain your reasoning.
- If no triples remain, return only the header row and separator row.
"""


def make_repair_args(args: argparse.Namespace, model_key: str) -> SimpleNamespace:
    config = REPAIR_MODEL_CONFIGS[model_key]
    return SimpleNamespace(
        model_cache=args.model_cache,
        model_id_or_path=config["model_id"],
        model_source="hf",
        model_type=config["model_type"],
        model_size=config["model_size"],
        disable_thinking=config["disable_thinking"],
        dtype=config["dtype"],
        tensor_parallel_size=args.repair_tp_override_map.get(model_key, config["tensor_parallel_size"]),
        trust_remote_code=False,
        gpu_memory_utilization=args.repair_gpu_memory_utilization,
        seed=args.repair_seed,
        max_model_len=args.repair_max_model_len,
        quantization="",
        enforce_eager=False,
        max_num_seqs=args.repair_max_num_seqs,
        batch_size=args.repair_batch_size,
        max_new_tokens=args.repair_max_new_tokens,
        temperature=args.repair_temperature,
        top_p=args.repair_top_p,
        top_k=args.repair_top_k,
        repetition_penalty=args.repair_repetition_penalty,
        debug_prompt_tokens=False,
        debug_prompt_token_count=16,
        debug_prompt_char_count=240,
    )


def parse_repair_output(raw_output: str) -> List[str]:
    return triples_to_strings(parse_triplets_markdown(raw_output))


def run_repair(
    *,
    samples: List[Dict[str, Any]],
    verifier_rows: List[Dict[str, Any]],
    repair_model: str,
    prompt_variant: str,
    args: argparse.Namespace,
    output_root: Path,
) -> Path:
    dataset = str(samples[0]["dataset"])
    baseline = str(samples[0]["baseline_name"])
    shot_k = int(samples[0].get("shot_k", 0))
    output_dir = ensure_dir(output_root / "markdown" / dataset)
    output_path = output_dir / repair_output_filename(dataset, baseline, repair_model, prompt_variant, shot_k)

    if output_path.exists():
        return output_path

    repair_records_dir = ensure_dir(output_root / "repair_records" / dataset / baseline / repair_model / prompt_variant)
    progress_jsonl_path = repair_records_dir / "records.jsonl"
    verifier_by_id = {row["sample_id"]: row for row in verifier_rows}
    flagged_samples = [sample for sample in samples if verifier_by_id[sample["sample_id"]]["flagged"]]
    repaired_by_id: Dict[str, List[str]] = {}
    repair_records: List[Dict[str, Any]] = []

    resumed_count = 0
    for record in load_jsonl_records(progress_jsonl_path):
        sample_id = str(record.get("sample_id", ""))
        parsed_triples = record.get("parsed_triples")
        if not sample_id or not isinstance(parsed_triples, list):
            continue
        repaired_by_id[sample_id] = [str(item) for item in parsed_triples]
        repair_records.append(record)
    resumed_count = len(repaired_by_id)

    repair_args = make_repair_args(args, repair_model)
    os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
    if repair_args.model_type == "qwen3" and os.environ.get("USE_HF") is None:
        os.environ["USE_HF"] = "1"

    pending = []
    for sample in flagged_samples:
        if sample["sample_id"] in repaired_by_id:
            continue
        feedback = verifier_by_id[sample["sample_id"]]["canonical_feedback"]
        messages = [
            {"role": "system", "content": REPAIR_SYSTEM},
            {
                "role": "user",
                "content": build_repair_user_prompt(
                    variant=prompt_variant,
                    source_text=sample["source_text"],
                    current_triples=sample["current_triples"],
                    canonical_feedback=feedback,
                ),
            },
        ]
        pending.append(
            {
                "sample_id": sample["sample_id"],
                "current_triples": sample["current_triples"],
                "messages": messages,
                "prompt": None,
            }
        )

    if resumed_count:
        print(
            f"[RESUME] dataset={dataset} baseline={baseline} repair={repair_model} "
            f"prompt={prompt_variant} resumed={resumed_count}/{len(flagged_samples)} flagged samples",
            flush=True,
        )

    configure_multiprocessing()
    set_seed(args.repair_seed)
    llm = None

    try:
        if pending:
            llm = load_vllm(repair_args, repair_args.model_id_or_path)
            tokenizer = llm.get_tokenizer()
            sampling_params = build_sampling_params(repair_args, tokenizer)
            for item in pending:
                item["prompt"] = render_prompt(repair_args, tokenizer, item["messages"])

        total_batches = math.ceil(len(pending) / args.repair_batch_size) if pending else 0
        progress_iter = enumerate(range(0, len(pending), args.repair_batch_size), start=1)
        if tqdm is not None:
            progress_iter = enumerate(
                tqdm(
                    range(0, len(pending), args.repair_batch_size),
                    total=total_batches,
                    desc=f"repair {dataset}/{baseline}/{repair_model}/{prompt_variant}",
                    dynamic_ncols=True,
                    leave=True,
                ),
                start=1,
            )
        for batch_idx, start in progress_iter:
            batch = pending[start : start + args.repair_batch_size]
            if tqdm is None and (batch_idx == 1 or batch_idx % 10 == 0 or batch_idx == total_batches):
                print(
                    f"[REPAIR-PROGRESS] dataset={dataset} baseline={baseline} "
                    f"repair={repair_model} prompt={prompt_variant} "
                    f"batch={batch_idx}/{total_batches} flagged={min(start + len(batch), len(pending))}/{len(pending)}",
                    flush=True,
                )
            outputs = llm.generate([item["prompt"] for item in batch], sampling_params=sampling_params, use_tqdm=False)
            for item, output in zip(batch, outputs):
                raw_output = output.outputs[0].text
                parsed_triples = parse_repair_output(raw_output)
                fell_back = False
                if not parsed_triples:
                    parsed_triples = list(item["current_triples"])
                    fell_back = True
                repaired_by_id[item["sample_id"]] = parsed_triples
                record = {
                    "sample_id": item["sample_id"],
                    "raw_model_output": raw_output,
                    "parsed_triples": parsed_triples,
                    "fell_back_to_previous": fell_back,
                    "messages": item["messages"],
                }
                repair_records.append(record)
                append_jsonl(progress_jsonl_path, record)
    finally:
        del llm
        gc.collect()
        verifier.clear_cuda_memory()

    entries = []
    changed = 0
    flagged = 0
    for sample in samples:
        verifier_row = verifier_by_id[sample["sample_id"]]
        output_triples = list(sample["current_triples"])
        raw_output = ""
        if verifier_row["flagged"]:
            flagged += 1
            output_triples = repaired_by_id.get(sample["sample_id"], list(sample["current_triples"]))
            raw_output = next(
                (record["raw_model_output"] for record in repair_records if record["sample_id"] == sample["sample_id"]),
                "",
            )
            if output_triples != sample["current_triples"]:
                changed += 1
        entries.append(
            {
                "sample_id": sample["sample_id"],
                "triplets": strings_to_dicts(output_triples),
                "raw_output": raw_output,
                "source_text": sample["source_text"],
                "baseline_name": baseline,
                "repair_model": repair_model,
                "prompt_variant": prompt_variant,
                "flagged": verifier_row["flagged"],
            }
        )

    payload = build_entries_payload(
        dataset=dataset,
        model_label=f"{repair_model}_{baseline}_{prompt_variant}",
        prompt_variant=prompt_variant,
        baseline=baseline,
        shot_k=shot_k,
        entries=entries,
        extra_meta={
            "repair_model": repair_model,
            "verifier_model": VERIFIER_MODEL_NAME,
            "verifier_prompt": "d2t_fixed",
        },
    )
    save_json(output_path, payload)
    repair_summary = {
        "dataset": dataset,
        "baseline_name": baseline,
        "repair_model": repair_model,
        "prompt_variant": prompt_variant,
        "verifier_model": VERIFIER_MODEL_NAME,
        "num_samples": len(samples),
        "flagged_samples": flagged,
        "changed_samples": changed,
    }
    save_json(repair_records_dir / "summary.json", repair_summary)
    save_json(repair_records_dir / "records.json", repair_records)
    if len(args.repair_models) == 1:
        model_root = ensure_dir(output_root / repair_model)
        model_output_path = ensure_dir(model_root / "markdown" / dataset) / output_path.name
        save_json(model_output_path, payload)
        model_records_dir = ensure_dir(model_root / "repair_records" / dataset / baseline / repair_model / prompt_variant)
        save_json(model_records_dir / "summary.json", repair_summary)
        save_json(model_records_dir / "records.json", repair_records)
        save_jsonl_rows(model_records_dir / "records.jsonl", repair_records)
    return output_path


def main() -> None:
    args = parse_args()
    output_root = ensure_dir(args.output_root)
    if args.input_json is not None and len(args.datasets) * len(args.baselines) != 1:
        raise ValueError("--input-json requires exactly one dataset and one baseline per invocation.")

    prepared: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for dataset in args.datasets:
        for baseline in args.baselines:
            samples = load_baseline_samples(
                dataset,
                baseline,
                args.max_samples,
                input_json=args.input_json,
                only_previously_flagged=args.only_previously_flagged,
            )
            if args.shot_k is not None:
                for sample in samples:
                    sample["shot_k"] = int(args.shot_k)
            prepared[(dataset, baseline)] = samples
            if args.stage in {"all", "prepare"} and args.input_json is None:
                baseline_path = save_baseline_payload(samples, output_root)
                print(f"[BASELINE] wrote {baseline_path}")

    verifier_cache: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    if args.stage in {"all", "verify", "repair"}:
        for dataset, baseline in prepared:
            verifier_rows = run_verifier(prepared[(dataset, baseline)], args, output_root)
            verifier_cache[(dataset, baseline)] = verifier_rows
            print(
                f"[VERIFY] dataset={dataset} baseline={baseline} "
                f"flagged={sum(1 for row in verifier_rows if row['flagged'])}/{len(verifier_rows)}"
            )

    if args.stage in {"all", "repair"}:
        for dataset, baseline in prepared:
            samples = prepared[(dataset, baseline)]
            verifier_rows = verifier_cache[(dataset, baseline)]
            for repair_model in args.repair_models:
                for prompt_variant in args.prompt_variants:
                    output_path = run_repair(
                        samples=samples,
                        verifier_rows=verifier_rows,
                        repair_model=repair_model,
                        prompt_variant=prompt_variant,
                        args=args,
                        output_root=output_root,
                    )
                    print(
                        f"[REPAIR] dataset={dataset} baseline={baseline} "
                        f"repair={repair_model} prompt={prompt_variant} -> {output_path}"
                    )


if __name__ == "__main__":
    main()
