#!/usr/bin/env python3
"""Build compact paper-facing T2D repair traces from full runtime outputs.

The compact release mirrors repair_d2t: shared inputs are stored once, while each
method JSONL stores only sparse per-round feedback plus the repaired triples.
Runtime prompts, source text, gold triples, raw model renderings, evaluator caches,
and mirrored per-model copies are intentionally omitted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional


HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = HERE / "t2d_repair_output"
DEFAULT_OUTPUT = HERE / "results" / "repair_outputs"

PAPER_MODELS = [
    "gemma3_4b",
    "gemma3_12b",
    "gemma3_27b",
    "llama3_1_8b",
    "llama3_3_70b_awq",
    "qwen3_4b",
    "qwen3_8b",
    "qwen3_14b",
    "qwen3_32b",
]

MODEL_IDS = {
    "gemma3_4b": "google/gemma-3-4b-it",
    "gemma3_12b": "google/gemma-3-12b-it",
    "gemma3_27b": "google/gemma-3-27b-it",
    "llama3_1_8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "llama3_3_70b_awq": "casperhansen/llama-3.3-70b-instruct-awq",
    "qwen3_4b": "Qwen/Qwen3-4B",
    "qwen3_8b": "Qwen/Qwen3-8B",
    "qwen3_14b": "Qwen/Qwen3-14B",
    "qwen3_32b": "Qwen/Qwen3-32B",
}

DATASETS = ("genwiki", "webnlg")
BASELINES = ("bt5", "regen")
METHOD_FILES = {
    "xqdt": "xqdt.jsonl",
    "factspotter": "factspotter.jsonl",
    "nli": "nli.jsonl",
    "pive": "pive.jsonl",
    "self_refine": "self_refine.jsonl",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Full historical t2d_repair_output directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Compact release output directory.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if isinstance(row, dict):
                    yield row


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def one_match(pattern: str) -> Path:
    matches = sorted(Path().glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one match for {pattern!r}, found {len(matches)}")
    return matches[0]


def require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def compact_common(
    dataset: str,
    baseline: str,
    model: str,
    round_id: int,
    row: Dict[str, Any],
    feedback: Any,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "dataset": dataset,
        "baseline": baseline,
        "model": model,
        "round": round_id,
        "sample_id": str(row["sample_id"]),
        "feedback": feedback,
    }
    if bool(row.get("fell_back_to_previous", False)):
        out["fell_back"] = True
    else:
        out["triples"] = [str(item) for item in row.get("parsed_triples", [])]
    return out


def xqdt_record_path(root: Path, round_id: int, dataset: str, baseline: str, model: str) -> Path:
    if round_id == 1:
        return require(
            root
            / "xqdt_output"
            / "round1"
            / "repair_records"
            / dataset
            / baseline
            / model
            / "g_bridge_recover_guard"
            / "records.jsonl"
        )
    return require(
        root
        / "xqdt_output"
        / f"round{round_id}"
        / model
        / "repair_records"
        / dataset
        / baseline
        / model
        / "g_bridge_recover_guard"
        / "records.jsonl"
    )


def xqdt_verifier_path(root: Path, round_id: int, dataset: str, baseline: str, model: str) -> Path:
    if round_id == 1:
        verifier_root = root / "xqdt_output" / "round1" / "verifier" / dataset / baseline
    else:
        verifier_root = root / "xqdt_output" / f"round{round_id}" / model / "verifier" / dataset / baseline
    matches = sorted(verifier_root.glob("*/verify_results.json"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one verifier result below {verifier_root}, found {len(matches)}")
    return matches[0]


def iter_xqdt(root: Path) -> Iterator[Dict[str, Any]]:
    # R1 feedback is shared before the repair models diverge. Cache it once per
    # dataset/baseline and attach only the canonical parsed feedback to records.
    feedback_cache: Dict[str, Dict[str, Any]] = {}
    for round_id in (1, 2, 3):
        for model in PAPER_MODELS:
            for dataset in DATASETS:
                for baseline in BASELINES:
                    verifier_path = xqdt_verifier_path(root, round_id, dataset, baseline, model)
                    cache_key = str(verifier_path)
                    if cache_key not in feedback_cache:
                        feedback_cache[cache_key] = {
                            str(item["sample_id"]): item.get("canonical_feedback", {})
                            for item in load_json(verifier_path)
                        }
                    feedback_by_id = feedback_cache[cache_key]
                    for row in read_jsonl(xqdt_record_path(root, round_id, dataset, baseline, model)):
                        sample_id = str(row["sample_id"])
                        feedback = feedback_by_id.get(sample_id)
                        if feedback is None:
                            raise KeyError(f"Missing XQDT feedback for {sample_id} in {verifier_path}")
                        yield compact_common(dataset, baseline, model, round_id, row, feedback)


def baseline_record_path(
    root: Path,
    round_id: int,
    dataset: str,
    baseline: str,
    model: str,
    mode: str,
) -> Path:
    if round_id == 1:
        return require(
            root
            / "baseline"
            / "round1"
            / "repair_records"
            / dataset
            / baseline
            / model
            / mode
            / "records.jsonl"
        )
    return require(
        root
        / "baseline"
        / f"round{round_id}_baseline"
        / model
        / "repair_records"
        / dataset
        / baseline
        / model
        / mode
        / "records.jsonl"
    )


def self_refine_record_path(
    root: Path,
    round_id: int,
    dataset: str,
    baseline: str,
    model: str,
) -> Path:
    if round_id == 1:
        return baseline_record_path(root, 1, dataset, baseline, model, "self_refine_v2")
    group_root = root / "baseline" / f"round{round_id}_self_refine"
    matches = sorted(
        group_root.glob(
            f"*_r{round_id}/repair_records/{dataset}/{baseline}/{model}/self_refine_v2/records.jsonl"
        )
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one self-refine record file for r{round_id}/{dataset}/{baseline}/{model}, "
            f"found {len(matches)}"
        )
    return matches[0]


def iter_factspotter(root: Path) -> Iterator[Dict[str, Any]]:
    for round_id in (1, 2, 3):
        for model in PAPER_MODELS:
            for dataset in DATASETS:
                for baseline in BASELINES:
                    path = baseline_record_path(root, round_id, dataset, baseline, model, "factspotter_binary")
                    for row in read_jsonl(path):
                        signal = row.get("signal_row") or {}
                        feedback = {"unsupported": [str(x) for x in signal.get("unsupported_triples", [])]}
                        yield compact_common(dataset, baseline, model, round_id, row, feedback)


def iter_nli(root: Path) -> Iterator[Dict[str, Any]]:
    for round_id in (1, 2, 3):
        for model in PAPER_MODELS:
            for dataset in DATASETS:
                for baseline in BASELINES:
                    path = baseline_record_path(root, round_id, dataset, baseline, model, "nli_binary")
                    for row in read_jsonl(path):
                        signal = row.get("signal_row") or {}
                        feedback: Dict[str, Any] = {
                            "unsupported": [str(x) for x in signal.get("unsupported_triples", [])],
                            "possible_missing": bool(signal.get("possible_missing", False)),
                        }
                        if signal.get("global_precision") is not None:
                            feedback["global_precision"] = float(signal["global_precision"])
                        yield compact_common(dataset, baseline, model, round_id, row, feedback)


def iter_pive(root: Path) -> Iterator[Dict[str, Any]]:
    for round_id in (1, 2, 3):
        for model in PAPER_MODELS:
            for dataset in DATASETS:
                for baseline in BASELINES:
                    path = baseline_record_path(root, round_id, dataset, baseline, model, "pive_missing")
                    for row in read_jsonl(path):
                        feedback = {"missing": [str(x) for x in row.get("missing_triples", [])]}
                        yield compact_common(dataset, baseline, model, round_id, row, feedback)


def iter_self_refine(root: Path) -> Iterator[Dict[str, Any]]:
    for round_id in (1, 2, 3):
        for model in PAPER_MODELS:
            for dataset in DATASETS:
                for baseline in BASELINES:
                    path = self_refine_record_path(root, round_id, dataset, baseline, model)
                    for row in read_jsonl(path):
                        feedback_text = str(row.get("raw_feedback_output", "")).strip()
                        used_repair = bool(row.get("used_repair_step", False))
                        # NO_ISSUES rows are the self-refine equivalent of an absent sparse
                        # record: they carry no actionable feedback and leave triples unchanged.
                        if not used_repair and feedback_text.upper() == "NO_ISSUES":
                            continue
                        out = compact_common(dataset, baseline, model, round_id, row, feedback_text)
                        if not used_repair:
                            out["used_repair_step"] = False
                        yield out


def manifest(counts: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "format": "xqdt-t2d-compact-repair-v1",
        "datasets": list(DATASETS),
        "baselines": list(BASELINES),
        "rounds": 3,
        "paper_models": MODEL_IDS,
        "shared_inputs": {
            "genwiki_unified": "../../t2d_repair_input/genwiki/genwiki_test_unified.json",
            "genwiki_bt5": "../../t2d_repair_input/genwiki/raw/genwiki_bt5.json",
            "genwiki_regen": "../../t2d_repair_input/genwiki/raw/genwiki_regen.json",
            "webnlg_unified": "../../t2d_repair_input/webnlg/webnlg_test_unified.json",
            "webnlg_bt5": "../../t2d_repair_input/webnlg/raw/webnlg_bt5.json",
            "webnlg_regen": "../../t2d_repair_input/webnlg/raw/webnlg_regen.json",
            "genwiki_reference_xml": "../../eval/data/genwiki_test_refs.xml",
            "webnlg_reference_xml": "../../eval/data/webnlg_test_refs.xml",
        },
        "semantics": (
            "Each method JSONL stores sparse per-round feedback and repaired triples. "
            "Initialize each dataset/baseline from the shared BT5/ReGen prediction array; "
            "apply matching records in ascending round order by sample_id. Samples absent "
            "from a round keep their previous triples. A record with fell_back=true also "
            "keeps the previous triples and therefore omits the redundant triples field. "
            "Source text, gold triples, prompts, raw rendered model output, verifier caches, "
            "evaluator intermediates, and mirrored runtime copies are intentionally not duplicated."
        ),
        "methods": {
            "xqdt": {
                "file": METHOD_FILES["xqdt"],
                "prompt_variant": "g_bridge_recover_guard",
                "feedback": "canonical missing/extra/incorrect verifier feedback",
            },
            "factspotter": {
                "file": METHOD_FILES["factspotter"],
                "mode": "factspotter_binary",
                "evaluator": "Inria-CEDAR/FactSpotter-DeBERTaV3-Base",
                "feedback": "unsupported triples used by the repair prompt",
            },
            "nli": {
                "file": METHOD_FILES["nli"],
                "mode": "nli_binary",
                "evaluator": "roberta-large-mnli",
                "feedback": "unsupported triples plus the coarse global missing-fact signal",
            },
            "pive": {
                "file": METHOD_FILES["pive"],
                "mode": "pive_missing",
                "verifier": "jiuzhouh/flan-t5-xxl-lora-verifier",
                "feedback": "missing triples used by the PiVe repair prompt",
            },
            "self_refine": {
                "files": {
                    "genwiki": "self_refine_genwiki.jsonl",
                    "webnlg": "self_refine_webnlg.jsonl",
                },
                "mode": "self_refine_v2",
                "feedback": "complete model-produced critique; NO_ISSUES unchanged rows omitted",
            },
        },
        "record_counts": counts,
    }


def main() -> None:
    args = parse_args()
    root = args.source_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    counts: Dict[str, Any] = {}
    iterators = {
        "xqdt": iter_xqdt,
        "factspotter": iter_factspotter,
        "nli": iter_nli,
        "pive": iter_pive,
    }
    for method, iterator in iterators.items():
        path = out_dir / METHOD_FILES[method]
        counts[method] = write_jsonl(path, iterator(root))
        print(f"[WRITE] {method}: {counts[method]} records -> {path}")

    stale_self_refine = out_dir / "self_refine.jsonl"
    if stale_self_refine.exists():
        stale_self_refine.unlink()

    self_refine_paths = {
        dataset: out_dir / f"self_refine_{dataset}.jsonl"
        for dataset in DATASETS
    }
    self_refine_counts = {dataset: 0 for dataset in DATASETS}
    handles = {
        dataset: path.open("w", encoding="utf-8")
        for dataset, path in self_refine_paths.items()
    }
    try:
        for row in iter_self_refine(root):
            dataset = str(row["dataset"])
            handles[dataset].write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            self_refine_counts[dataset] += 1
    finally:
        for handle in handles.values():
            handle.close()
    counts["self_refine"] = self_refine_counts
    for dataset in DATASETS:
        print(
            f"[WRITE] self_refine/{dataset}: {self_refine_counts[dataset]} records "
            f"-> {self_refine_paths[dataset]}"
        )

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest(counts), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[WRITE] manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
