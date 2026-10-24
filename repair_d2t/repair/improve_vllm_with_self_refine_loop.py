#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Iterative D2T repair loop driven by model-internal self-refine feedback."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

from d2t_baseline_prompts import (
    SELF_REFINE_SYSTEM,
    Triple,
    build_self_refine_markdown_prompt,
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
    save_generation_snapshot,
    save_json,
    triples_to_strings,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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

    parser.add_argument("--model_cache", type=str, default=".cache")
    return parser.parse_args()


def make_triples(source_triples: List[str]) -> List[Triple]:
    triples: List[Triple] = []
    for triple in source_triples:
        parts = [part.strip() for part in str(triple).split("|")]
        if len(parts) != 3:
            continue
        triples.append(Triple(parts[0], parts[1], parts[2]))
    return triples


def parse_self_refine_output(raw_text: str, previous_text: str) -> Tuple[str, str]:
    text = raw_text.strip()
    if not text:
        return previous_text, "empty_fallback"

    patterns = [
        r"(?is)##\s*revised text\s*(.+)$",
        r"(?is)#\s*revised text\s*(.+)$",
        r"(?is)revised text\s*:?\s*(.+)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        candidate = match.group(1).strip()
        candidate = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", candidate).strip()
        candidate = re.sub(r"\n?```$", "", candidate).strip()
        if candidate:
            return candidate, "revised_section"

    # If the model skipped the markdown sections and returned plain text,
    # accept it only when it does not look like the issue table.
    if "## factual issues" not in text.lower() and "| type |" not in text.lower():
        return text, "plain_text_fallback"
    return previous_text, "section_missing_fallback"


def run_self_refine_round(
    args: argparse.Namespace,
    data: Dict[str, Any],
    current_texts: Dict[int, str],
    active_indices: List[int],
    round_idx: int,
    output_dir: Path,
) -> tuple[Dict[int, str], List[Dict[str, Any]]]:
    from generate_vllm_simple import build_sampling_params, load_vllm, render_prompt
    from generate_vllm_simple import configure_multiprocessing, set_seed

    if not active_indices:
        save_json(output_dir / f"round{round_idx}_self_refine_records.json", [])
        return {}, []

    repair_args = make_repair_args(args)
    os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
    if repair_args.model_type == "qwen3" and os.environ.get("USE_HF") is None:
        os.environ["USE_HF"] = "1"

    configure_multiprocessing()
    set_seed(args.repair_seed)

    llm = None
    repaired_texts: Dict[int, str] = {}
    records: List[Dict[str, Any]] = []

    try:
        llm = load_vllm(repair_args, args.generator_model_id_or_path)
        tokenizer = llm.get_tokenizer()
        sampling_params = build_sampling_params(repair_args, tokenizer)

        pending = []
        for idx in tqdm(active_indices, desc=f"self_refine r{round_idx} build"):
            entry = data["entries"][idx]
            source_triples = triples_to_strings(entry)
            user_text = build_self_refine_markdown_prompt(
                make_triples(source_triples),
                current_texts[idx],
            )
            messages = [
                {"role": "system", "content": SELF_REFINE_SYSTEM},
                {"role": "user", "content": user_text},
            ]
            prompt = render_prompt(repair_args, tokenizer, messages)
            pending.append({"idx": idx, "messages": messages, "prompt": prompt, "source_triples": source_triples})

        for start in tqdm(range(0, len(pending), args.repair_batch_size), desc=f"self_refine r{round_idx}"):
            batch = pending[start : start + args.repair_batch_size]
            outputs = llm.generate([item["prompt"] for item in batch], sampling_params=sampling_params, use_tqdm=False)
            for item, output in zip(batch, outputs):
                raw = output.outputs[0].text
                repaired_text, parse_mode = parse_self_refine_output(raw, current_texts[item["idx"]])
                changed = repaired_text.strip() != current_texts[item["idx"]].strip()
                repaired_texts[item["idx"]] = repaired_text
                records.append(
                    {
                        "index": item["idx"],
                        "sample_id": get_entry_key(data["entries"][item["idx"]], item["idx"]),
                        "round": round_idx,
                        "previous_text": current_texts[item["idx"]],
                        "repaired_text": repaired_text,
                        "changed": changed,
                        "raw_model_output": raw,
                        "parse_mode": parse_mode,
                        "source_triples": item["source_triples"],
                        "messages": item["messages"],
                    }
                )
    finally:
        del llm
        clear_cuda_memory()

    save_json(output_dir / f"round{round_idx}_self_refine_records.json", records)
    return repaired_texts, records


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

    save_json(output_dir / "round0_baseline.json", data)
    save_generation_snapshot(args, data, current_texts, output_dir, 0)

    run_summary = {
        "baseline_json": str(args.baseline_json),
        "output_dir": str(output_dir),
        "generator_model_id_or_path": args.generator_model_id_or_path,
        "repair_temperature": args.repair_temperature,
        "repair_seed": args.repair_seed,
        "self_refine_format": "markdown_table",
        "rounds": [],
    }
    active_indices = list(range(len(data["entries"])))

    for round_idx in range(1, args.max_rounds + 1):
        snapshot_path = output_dir / f"round{round_idx}_improved.json"
        records_path = output_dir / f"round{round_idx}_self_refine_records.json"
        if snapshot_path.exists() and records_path.exists():
            try:
                snap_data = load_json(snapshot_path)
                records = load_json(records_path)
                if isinstance(snap_data, dict) and isinstance(records, list):
                    for idx in range(len(data["entries"])):
                        text = get_text_from_entry(snap_data["entries"][idx], args.lang_code)
                        if text:
                            current_texts[idx] = text
                    changed_indices = [
                        int(row["index"])
                        for row in records
                        if isinstance(row, dict) and row.get("changed") and "index" in row
                    ]
                    active_indices = changed_indices
                    run_summary["rounds"].append(
                        {
                            "round": round_idx,
                            "processed": len(records),
                            "changed": len(changed_indices),
                            "unchanged": len(records) - len(changed_indices),
                            "active_for_next_round": len(changed_indices),
                            "resumed": True,
                        }
                    )
                    continue
            except Exception:
                pass

        repaired_texts, records = run_self_refine_round(
            args=args,
            data=data,
            current_texts=current_texts,
            active_indices=active_indices,
            round_idx=round_idx,
            output_dir=output_dir,
        )
        current_texts.update(repaired_texts)
        changed_indices = [
            int(row["index"])
            for row in records
            if isinstance(row, dict) and row.get("changed") and "index" in row
        ]
        save_generation_snapshot(args, data, current_texts, output_dir, round_idx)
        run_summary["rounds"].append(
            {
                "round": round_idx,
                "processed": len(records),
                "changed": len(changed_indices),
                "unchanged": len(records) - len(changed_indices),
                "active_for_next_round": len(changed_indices),
            }
        )
        active_indices = changed_indices

    save_json(output_dir / "run_summary.json", run_summary)
    print(json.dumps(run_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
