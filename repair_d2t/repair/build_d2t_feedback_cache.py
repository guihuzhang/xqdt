#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build cached feedback rows for D2T repair baselines."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scoring"))

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import eval_vllm_ref_less as evaluator
from evaluate_repair_text_metrics import (
    linearize_predicate_for_factspotter,
    load_nli_model,
    load_nli_templates,
    nli_entailment_probs,
    resolve_nli_template_file,
    triple_to_nli_template,
)
from improve_vllm_with_eval_loop import (
    ensure_dir,
    ensure_file,
    get_entry_key,
    get_text_from_entry,
    load_json,
    normalize_generation_payload,
    save_json,
    triples_to_strings,
)


def build_nli_feedback_rows(
    *,
    entries: Sequence[Dict[str, Any]],
    current_texts: Dict[int, str],
    active_indices: Optional[Sequence[int]] = None,
    dataset_name: str,
    baseline_name: str,
    round_idx: int,
    lang_code: str,
    metrics_dir: Path,
    entailment_threshold: float,
    global_warning_threshold: float,
    include_global_warning: bool,
) -> List[Dict[str, Any]]:
    bundle = load_nli_model()
    if bundle is None:
        raise RuntimeError("Failed to load NLI model bundle.")
    tokenizer, model, device = bundle

    template_file = resolve_nli_template_file(metrics_dir)
    templates = load_nli_templates(str(template_file)) if template_file is not None else {}

    indices = list(active_indices) if active_indices is not None else list(range(len(entries)))

    per_fact_pairs: List[Tuple[str, str]] = []
    per_fact_meta: List[Tuple[int, str]] = []
    global_pairs: List[Tuple[str, str]] = []
    global_meta: List[int] = []
    source_triples_by_idx: Dict[int, List[str]] = {}

    for idx in indices:
        entry = entries[idx]
        source_triples = triples_to_strings(entry)
        source_triples_by_idx[idx] = source_triples
        hyp = current_texts[idx].strip()
        template_sents = [
            template
            for template in (triple_to_nli_template(triple, templates) for triple in source_triples)
            if template
        ]
        for triple, template in zip(source_triples, template_sents):
            per_fact_pairs.append((hyp, template))
            per_fact_meta.append((idx, triple))
        if hyp and template_sents:
            global_pairs.append((" ".join(template_sents), hyp))
            global_meta.append(idx)

    per_fact_scores = nli_entailment_probs(tokenizer, model, device, per_fact_pairs, batch_size=16)
    global_scores = nli_entailment_probs(tokenizer, model, device, global_pairs, batch_size=8)

    flagged_by_idx: Dict[int, List[str]] = {idx: [] for idx in indices}
    for (idx, triple), score in zip(per_fact_meta, per_fact_scores):
        if float(score) < entailment_threshold:
            flagged_by_idx[idx].append(triple)

    global_warning_by_idx: Dict[int, str] = {idx: "" for idx in indices}
    if include_global_warning:
        for idx, score in zip(global_meta, global_scores):
            if float(score) < global_warning_threshold:
                global_warning_by_idx[idx] = (
                    "The text may contain unsupported content not grounded in the source triples."
                )

    rows: List[Dict[str, Any]] = []
    for idx in indices:
        entry = entries[idx]
        parsed_output = {
            "missing": [],
            "extra": [],
            "incorrect": list(flagged_by_idx[idx]),
        }
        global_warning = global_warning_by_idx[idx]
        perfect = (not parsed_output["incorrect"]) and (not global_warning)
        metrics = evaluator.compute_prf(parsed_output, len(source_triples_by_idx[idx]))
        rows.append(
            {
                "sample_id": get_entry_key(entry, idx),
                "index": idx,
                "round": round_idx,
                "dataset_name": dataset_name,
                "baseline_name": baseline_name,
                "input_text": current_texts[idx],
                "source_triples": source_triples_by_idx[idx],
                "raw_source_triples": entry.get("modifiedtripleset", []),
                "prompt": "",
                "model_output": "",
                "parsed_output": parsed_output,
                "feedback_payload": {
                    "facts_to_check": list(flagged_by_idx[idx]),
                    "global_extra_warning": global_warning,
                },
                "metrics": metrics,
                "perfect": perfect,
            }
        )
    return rows


_FACTSPOTTER_BUNDLE: Optional[Tuple[Any, Any, Any]] = None


def load_factspotter_bundle() -> Tuple[Any, Any, Any]:
    global _FACTSPOTTER_BUNDLE
    if _FACTSPOTTER_BUNDLE is not None:
        return _FACTSPOTTER_BUNDLE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("Inria-CEDAR/FactSpotter-DeBERTaV3-Base")
    model = AutoModelForSequenceClassification.from_pretrained(
        "Inria-CEDAR/FactSpotter-DeBERTaV3-Base"
    ).to(device)
    model.eval()
    _FACTSPOTTER_BUNDLE = (tokenizer, model, device)
    return _FACTSPOTTER_BUNDLE


def linearize_factspotter_triple(triple: str) -> Optional[str]:
    parts = [part.strip().strip('"') for part in str(triple).split(" | ")]
    if len(parts) != 3:
        parts = [part.strip().strip('"') for part in str(triple).split("|")]
    if len(parts) != 3:
        return None
    subject, predicate, obj = parts
    return f"{subject}, {linearize_predicate_for_factspotter(predicate)}, {obj}".replace("_", " ")


def factspotter_support_probs(
    pairs: Sequence[Tuple[str, str]],
    batch_size: int = 8,
) -> List[float]:
    tokenizer, model, device = load_factspotter_bundle()
    scores: List[float] = []
    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            texts = [item[0] for item in batch]
            triples = [item[1] for item in batch]
            encoded = tokenizer(texts, triples, truncation=True, padding=True, return_tensors="pt").to(device)
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=1)
            scores.extend(float(item[0]) for item in probs)
    return scores


def build_factspotter_feedback_rows(
    *,
    entries: Sequence[Dict[str, Any]],
    current_texts: Dict[int, str],
    active_indices: Optional[Sequence[int]] = None,
    dataset_name: str,
    baseline_name: str,
    round_idx: int,
    support_threshold: float,
) -> List[Dict[str, Any]]:
    indices = list(active_indices) if active_indices is not None else list(range(len(entries)))
    pairs: List[Tuple[str, str]] = []
    meta: List[Tuple[int, str]] = []
    source_triples_by_idx: Dict[int, List[str]] = {}
    for idx in indices:
        entry = entries[idx]
        source_triples = triples_to_strings(entry)
        source_triples_by_idx[idx] = source_triples
        hyp = current_texts[idx].strip()
        for triple in source_triples:
            linearized = linearize_factspotter_triple(triple)
            if linearized is None:
                continue
            pairs.append((hyp, linearized))
            meta.append((idx, triple))

    probs = factspotter_support_probs(pairs, batch_size=8)
    flagged_by_idx: Dict[int, List[str]] = {idx: [] for idx in indices}
    for (idx, triple), score in zip(meta, probs):
        if float(score) < support_threshold:
            flagged_by_idx[idx].append(triple)

    rows: List[Dict[str, Any]] = []
    for idx in indices:
        entry = entries[idx]
        parsed_output = {
            "missing": [],
            "extra": [],
            "incorrect": list(flagged_by_idx[idx]),
        }
        perfect = not parsed_output["incorrect"]
        metrics = evaluator.compute_prf(parsed_output, len(source_triples_by_idx[idx]))
        rows.append(
            {
                "sample_id": get_entry_key(entry, idx),
                "index": idx,
                "round": round_idx,
                "dataset_name": dataset_name,
                "baseline_name": baseline_name,
                "input_text": current_texts[idx],
                "source_triples": source_triples_by_idx[idx],
                "raw_source_triples": entry.get("modifiedtripleset", []),
                "prompt": "",
                "model_output": "",
                "parsed_output": parsed_output,
                "feedback_payload": {
                    "facts_to_check": list(flagged_by_idx[idx]),
                },
                "metrics": metrics,
                "perfect": perfect,
            }
        )
    return rows


def build_summary(rows: Sequence[Dict[str, Any]], mode: str, round_idx: int) -> Dict[str, Any]:
    perfect = sum(1 for row in rows if row.get("perfect"))
    imperfect = len(rows) - perfect
    summary = {
        "round": round_idx,
        "judged": len(rows),
        "perfect": perfect,
        "imperfect": imperfect,
        "evaluator_model": mode,
        "summary": evaluator.summarize_results(list(rows)) if rows else {},
    }
    if mode == "nli":
        summary["global_extra_warning_count"] = sum(
            1 for row in rows if row.get("feedback_payload", {}).get("global_extra_warning")
        )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["nli", "factspotter"], required=True)
    parser.add_argument("--input_json", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--summary_json", type=Path, default=None)
    parser.add_argument("--dataset_name", type=str, default="")
    parser.add_argument("--baseline_name", type=str, default="")
    parser.add_argument("--round_idx", type=int, default=1)
    parser.add_argument("--lang_code", type=str, default="en")
    parser.add_argument("--metrics_dir", type=Path, default=Path(__file__).resolve().parent.parent / "metrics")
    parser.add_argument("--nli_entailment_threshold", type=float, default=0.5)
    parser.add_argument("--nli_global_warning_threshold", type=float, default=0.5)
    parser.add_argument("--nli_disable_global_warning", action="store_true")
    parser.add_argument("--factspotter_support_threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_json = ensure_file(args.input_json)
    output_json = args.output_json.expanduser().resolve()
    summary_json = (
        args.summary_json.expanduser().resolve()
        if args.summary_json
        else output_json.with_name(output_json.name.replace("_active.json", "_summary.json"))
    )

    payload = normalize_generation_payload(load_json(input_json), args.lang_code)
    entries = payload["entries"]
    current_texts = {idx: get_text_from_entry(entry, args.lang_code) for idx, entry in enumerate(entries)}

    if args.mode == "nli":
        rows = build_nli_feedback_rows(
            entries=entries,
            current_texts=current_texts,
            dataset_name=args.dataset_name,
            baseline_name=args.baseline_name,
            round_idx=args.round_idx,
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
            dataset_name=args.dataset_name,
            baseline_name=args.baseline_name,
            round_idx=args.round_idx,
            support_threshold=args.factspotter_support_threshold,
        )

    save_json(output_json, rows)
    save_json(summary_json, build_summary(rows, args.mode, args.round_idx))
    print(json.dumps({"output_json": str(output_json), "summary_json": str(summary_json), "num_rows": len(rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
