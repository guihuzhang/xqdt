#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Iterative D2T repair loop driven by cached NLI / FactSpotter feedback."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence

from tqdm import tqdm

from build_d2t_feedback_cache import (
    build_factspotter_feedback_rows,
    build_nli_feedback_rows,
    build_summary,
)
from d2t_baseline_prompts import (
    FACTSPOTTER_REVISION_SYSTEM,
    NLI_REVISION_SYSTEM,
    Triple,
    build_factspotter_revision_prompt,
    build_nli_revision_prompt,
)
from improve_vllm_with_eval_loop import (
    clear_cuda_memory,
    ensure_dir,
    ensure_file,
    get_entry_key,
    get_text_from_entry,
    load_json,
    make_repair_args,
    normalize_generation_payload,
    parse_prediction_result,
    save_cumulative_eval,
    save_generation_snapshot,
    save_json,
    triples_to_strings,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback_mode", choices=["nli", "factspotter"], required=True)
    parser.add_argument("--baseline_json", type=Path, required=True)
    parser.add_argument("--fewshot_source", type=Path, default=None, help="Legacy compatibility option; unused by the paper single-context baseline runners.")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_rounds", type=int, default=3)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--dataset_name", type=str, default="")
    parser.add_argument("--baseline_name", type=str, default="")
    parser.add_argument("--lang_code", type=str, default="en")
    parser.add_argument("--target_lang", type=str, default="en")
    parser.add_argument("--fewshot_lang", type=str, default="en")
    parser.add_argument("--fewshot_style", type=str, default="overlap")
    parser.add_argument("--generation_technique", type=str, default="dir")
    parser.add_argument("--descriptions", type=str, default="none")
    parser.add_argument("--labels", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=4)

    parser.add_argument("--generator_model_id_or_path", type=str, required=True)
    parser.add_argument("--generator_model_source", type=str, default="hf")
    parser.add_argument("--generator_model_type", type=str, default="llama3")
    parser.add_argument("--generator_model_size", type=str, default="8b")
    parser.add_argument("--disable_thinking", action="store_true")
    parser.add_argument("--repair_temperature", type=float, default=0.7)
    parser.add_argument("--repair_seed", type=int, default=42)
    parser.add_argument("--repair_top_p", type=float, default=0.7)
    parser.add_argument("--repair_top_k", type=int, default=50)
    parser.add_argument("--repair_repetition_penalty", type=float, default=1.0)
    parser.add_argument("--repair_max_new_tokens", type=int, default=1280)
    parser.add_argument("--repair_batch_size", type=int, default=4)
    parser.add_argument("--generator_dtype", type=str, default="bfloat16")
    parser.add_argument("--generator_tensor_parallel_size", type=int, default=1)
    parser.add_argument("--generator_gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--generator_max_model_len", type=int, default=4096)
    parser.add_argument("--generator_max_num_seqs", type=int, default=None)
    parser.add_argument("--generator_quantization", type=str, default="")
    parser.add_argument("--generator_trust_remote_code", action="store_true")
    parser.add_argument("--generator_enforce_eager", action="store_true")

    parser.add_argument("--reuse_round1_eval_json", type=Path, default=None)
    parser.add_argument("--model_cache", type=str, default=".cache")
    parser.add_argument("--nli_entailment_threshold", type=float, default=0.5)
    parser.add_argument("--nli_global_warning_threshold", type=float, default=0.5)
    parser.add_argument("--nli_disable_global_warning", action="store_true")
    parser.add_argument("--factspotter_support_threshold", type=float, default=0.5)
    parser.add_argument("--metrics_dir", type=Path, default=Path(__file__).resolve().parent.parent / "metrics")
    return parser.parse_args()


def feedback_rows_for_round(
    *,
    args: argparse.Namespace,
    data: Dict[str, Any],
    current_texts: Dict[int, str],
    active_indices: Sequence[int],
    round_idx: int,
    output_dir: Path,
) -> List[Dict[str, Any]]:
    local_eval_path = output_dir / f"round{round_idx}_eval_active.json"
    local_summary_path = output_dir / f"round{round_idx}_eval_summary.json"
    expected_indices = sorted(int(idx) for idx in active_indices)
    if local_eval_path.is_file():
        rows = load_json(local_eval_path)
        if isinstance(rows, list):
            row_indices = sorted(
                int(row["index"])
                for row in rows
                if isinstance(row, dict) and "index" in row
            )
            if row_indices == expected_indices:
                return rows

    if not active_indices:
        save_json(local_eval_path, [])
        save_json(local_summary_path, build_summary([], args.feedback_mode, round_idx))
        return []

    if round_idx == 1 and args.reuse_round1_eval_json:
        reuse_path = ensure_file(args.reuse_round1_eval_json)
        rows = load_json(reuse_path)
        if not isinstance(rows, list):
            raise ValueError(f"Expected a list in reused round1 eval json: {reuse_path}")
        rows = [
            row
            for row in rows
            if isinstance(row, dict) and int(row.get("index", -1)) in set(expected_indices)
        ]
        save_json(local_eval_path, rows)
        reuse_summary = args.reuse_round1_eval_json.with_name(
            args.reuse_round1_eval_json.name.replace("_active.json", "_summary.json")
        )
        if reuse_summary.is_file():
            save_json(local_summary_path, build_summary(rows, args.feedback_mode, round_idx))
        else:
            save_json(local_summary_path, build_summary(rows, args.feedback_mode, round_idx))
        return rows

    entries = data["entries"]
    if args.feedback_mode == "nli":
        rows = build_nli_feedback_rows(
            entries=entries,
            current_texts=current_texts,
            active_indices=active_indices,
            dataset_name=args.dataset_name,
            baseline_name=args.baseline_name,
            round_idx=round_idx,
            lang_code=args.lang_code,
            metrics_dir=args.metrics_dir.expanduser().resolve(),
            entailment_threshold=args.nli_entailment_threshold,
            global_warning_threshold=args.nli_global_warning_threshold,
            include_global_warning=not args.nli_disable_global_warning,
        )
    else:
        rows = build_factspotter_feedback_rows(
            entries=entries,
            current_texts=current_texts,
            active_indices=active_indices,
            dataset_name=args.dataset_name,
            baseline_name=args.baseline_name,
            round_idx=round_idx,
            support_threshold=args.factspotter_support_threshold,
        )

    save_json(local_eval_path, rows)
    save_json(local_summary_path, build_summary(rows, args.feedback_mode, round_idx))
    return rows


def make_triples(source_triples: Sequence[str]) -> List[Triple]:
    triples: List[Triple] = []
    for triple in source_triples:
        parts = [part.strip() for part in str(triple).split("|")]
        if len(parts) != 3:
            continue
        triples.append(Triple(parts[0], parts[1], parts[2]))
    return triples


def build_signal_prompt(
    *,
    args: argparse.Namespace,
    source_triples: Sequence[str],
    previous_text: str,
    row: Dict[str, Any],
) -> tuple[str, str]:
    feedback_payload = row.get("feedback_payload", {}) if isinstance(row, dict) else {}
    triples = make_triples(source_triples)
    facts_to_check = make_triples(feedback_payload.get("facts_to_check", row.get("parsed_output", {}).get("incorrect", [])))
    if args.feedback_mode == "nli":
        return (
            NLI_REVISION_SYSTEM,
            build_nli_revision_prompt(
                triples,
                previous_text,
                facts_to_check,
                feedback_payload.get("global_extra_warning") or None,
            ),
        )
    return (
        FACTSPOTTER_REVISION_SYSTEM,
        build_factspotter_revision_prompt(
            triples,
            previous_text,
            facts_to_check,
        ),
    )


def run_repair_round(
    args: argparse.Namespace,
    data: Dict[str, Any],
    current_texts: Dict[int, str],
    eval_rows: List[Dict[str, Any]],
    round_idx: int,
    output_dir: Path,
) -> Dict[int, str]:
    eval_by_idx = {row["index"]: row for row in eval_rows if not row.get("perfect")}
    repair_indices = sorted(eval_by_idx)
    if not repair_indices:
        return {}

    from generate_vllm_simple import build_sampling_params, load_vllm, render_prompt
    from generate_vllm_simple import configure_multiprocessing, set_seed

    repair_args = make_repair_args(args)
    os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
    if repair_args.model_type == "qwen3" and os.environ.get("USE_HF") is None:
        os.environ["USE_HF"] = "1"

    configure_multiprocessing()
    set_seed(args.repair_seed)

    llm = None
    repaired_texts: Dict[int, str] = {}
    repair_records: List[Dict[str, Any]] = []

    try:
        llm = load_vllm(repair_args, args.generator_model_id_or_path)
        tokenizer = llm.get_tokenizer()
        sampling_params = build_sampling_params(repair_args, tokenizer)

        pending = []
        for idx in tqdm(repair_indices, desc=f"repair r{round_idx} build"):
            entry = data["entries"][idx]
            source_triples = triples_to_strings(entry)
            system_text, user_text = build_signal_prompt(
                args=args,
                source_triples=source_triples,
                previous_text=current_texts[idx],
                row=eval_by_idx[idx],
            )
            messages = [
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_text},
            ]
            prompt = render_prompt(repair_args, tokenizer, messages)
            pending.append({"idx": idx, "messages": messages, "prompt": prompt})

        for start in tqdm(range(0, len(pending), args.repair_batch_size), desc=f"repair r{round_idx}"):
            batch = pending[start : start + args.repair_batch_size]
            outputs = llm.generate([item["prompt"] for item in batch], sampling_params=sampling_params, use_tqdm=False)
            for item, output in zip(batch, outputs):
                raw = output.outputs[0].text
                repaired_text, parse_mode = parse_prediction_result(raw)
                fell_back_to_previous = False
                if not repaired_text:
                    repaired_text = current_texts[item["idx"]]
                    parse_mode = "empty_fallback"
                    fell_back_to_previous = True
                repaired_texts[item["idx"]] = repaired_text
                repair_records.append(
                    {
                        "index": item["idx"],
                        "sample_id": get_entry_key(data["entries"][item["idx"]], item["idx"]),
                        "round": round_idx,
                        "previous_text": current_texts[item["idx"]],
                        "repaired_text": repaired_text,
                        "raw_model_output": raw,
                        "parse_mode": parse_mode,
                        "fell_back_to_previous": fell_back_to_previous,
                        "feedback": eval_by_idx[item["idx"]],
                        "messages": item["messages"],
                    }
                )
    finally:
        del llm
        clear_cuda_memory()

    save_json(output_dir / f"round{round_idx}_repair_records.json", repair_records)
    return repaired_texts


def main() -> None:
    args = parse_args()
    os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # Reuse the existing repair arg builder, which expects these long-form names.
    args.target_language = args.target_lang
    args.fewshot_language = args.fewshot_lang
    args.repair_feedback_format = "json"
    args.repair_context_mode = "single"
    args.repair_history_fewshot_selector = "lightweight"

    args.baseline_json = ensure_file(args.baseline_json)
    output_dir = ensure_dir(args.output_dir)

    data = normalize_generation_payload(load_json(args.baseline_json), args.lang_code)
    if args.max_samples is not None:
        data = copy.deepcopy(data)
        data["entries"] = data["entries"][: args.max_samples]

    current_texts = {idx: get_text_from_entry(entry, args.lang_code) for idx, entry in enumerate(data["entries"])}
    all_eval_by_idx: Dict[int, Dict[str, Any]] = {}
    active_indices = list(range(len(data["entries"])))

    save_json(output_dir / "round0_baseline.json", data)
    save_generation_snapshot(args, data, current_texts, output_dir, 0)

    run_summary = {
        "baseline_json": str(args.baseline_json),
        "output_dir": str(output_dir),
        "feedback_mode": args.feedback_mode,
        "reuse_round1_eval_json": str(args.reuse_round1_eval_json) if args.reuse_round1_eval_json else None,
        "generator_model_id_or_path": args.generator_model_id_or_path,
        "repair_temperature": args.repair_temperature,
        "repair_seed": args.repair_seed,
        "rounds": [],
    }

    for round_idx in range(1, args.max_rounds + 1):
        snapshot_path = output_dir / f"round{round_idx}_improved.json"
        cum_eval_path = output_dir / f"round{round_idx}_eval.json"
        if snapshot_path.exists() and cum_eval_path.exists():
            try:
                snap_data = load_json(snapshot_path)
                cum_rows = load_json(cum_eval_path)
                if isinstance(snap_data, dict) and isinstance(cum_rows, list):
                    for idx in range(len(data["entries"])):
                        text = get_text_from_entry(snap_data["entries"][idx], args.lang_code)
                        if text:
                            current_texts[idx] = text
                    for row in cum_rows:
                        if isinstance(row, dict) and "index" in row:
                            all_eval_by_idx[int(row["index"])] = row
                    next_imperfect = [int(row["index"]) for row in cum_rows if not row.get("perfect")]
                    run_summary["rounds"].append(
                        {
                            "round": round_idx,
                            "judged": len(cum_rows),
                            "new_perfect": sum(1 for r in cum_rows if r.get("perfect")),
                            "remaining_imperfect": len(next_imperfect),
                            "resumed": True,
                        }
                    )
                    if not next_imperfect:
                        break
                    active_indices = next_imperfect
                    continue
            except Exception:
                pass

        eval_rows = feedback_rows_for_round(
            args=args,
            data=data,
            current_texts=current_texts,
            active_indices=active_indices,
            round_idx=round_idx,
            output_dir=output_dir,
        )
        perfect_indices = [row["index"] for row in eval_rows if row.get("perfect")]
        imperfect_indices = [row["index"] for row in eval_rows if not row.get("perfect")]
        for row in eval_rows:
            all_eval_by_idx[row["index"]] = row
        save_cumulative_eval(output_dir, round_idx, all_eval_by_idx)
        run_summary["rounds"].append(
            {
                "round": round_idx,
                "judged": len(eval_rows),
                "new_perfect": len(perfect_indices),
                "remaining_imperfect": len(imperfect_indices),
            }
        )

        if not imperfect_indices:
            save_generation_snapshot(args, data, current_texts, output_dir, round_idx)
            break

        repaired_texts = run_repair_round(
            args=args,
            data=data,
            current_texts=current_texts,
            eval_rows=eval_rows,
            round_idx=round_idx,
            output_dir=output_dir,
        )
        current_texts.update(repaired_texts)
        save_generation_snapshot(args, data, current_texts, output_dir, round_idx)
        active_indices = imperfect_indices

    save_json(output_dir / "run_summary.json", run_summary)
    print(json.dumps(run_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
