#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Iterative vLLM D2T repair using LoRA reference-less evaluators.

The loop is:
1. read an existing generation JSON file,
2. judge current texts with the matching LoRA evaluator,
3. repair only imperfect samples with the original generation model,
4. repeat on the remaining imperfect samples.

Judge defaults are intentionally stable: temperature=0.3, seed=2023.
Repair defaults match the baseline generation setup: temperature=0.7, seed=42.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm

import eval_vllm_ref_less as evaluator
from prompt_variants_v1 import (
    DEFAULT_REPAIR_PROMPT_VARIANT,
    VARIANT_ORDER as REPAIR_PROMPT_VARIANTS,
    get_prompt_variant,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SYSTEM = "You are a careful data-to-text generation assistant."
REPAIR_SYSTEM = (
    "You are a data-to-text revision assistant. Preserve correct content while "
    "correcting flagged errors."
)


def ensure_file(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    return path


def ensure_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_source_triple(triple: Any) -> Optional[Dict[str, str]]:
    if isinstance(triple, dict):
        subject = str(triple.get("subject", "")).strip()
        predicate = str(triple.get("property", triple.get("predicate", ""))).strip()
        obj = str(triple.get("object", "")).strip()
    else:
        parts = [part.strip() for part in str(triple).split(" | ")]
        if len(parts) != 3:
            parts = [part.strip() for part in str(triple).split("|")]
        if len(parts) != 3:
            return None
        subject, predicate, obj = parts
    if not subject or not predicate or not obj:
        return None
    return {"subject": subject, "property": predicate, "object": obj}


def normalize_generation_payload(payload: Any, lang_code: str) -> Dict[str, Any]:
    if isinstance(payload, dict) and isinstance(payload.get("entries"), list):
        return payload

    if not isinstance(payload, list):
        raise ValueError("Expected generation JSON with an 'entries' list or an eval_input JSON list.")

    entries: List[Dict[str, Any]] = []
    for idx, sample in enumerate(payload):
        if not isinstance(sample, dict):
            raise ValueError(f"Malformed eval_input sample at index {idx}")
        source_triples = sample.get("source_triples") or sample.get("raw_source_triples") or []
        modified_triples = [parsed for triple in source_triples if (parsed := parse_source_triple(triple))]
        entry = copy.deepcopy(sample)
        entry["modifiedtripleset"] = modified_triples
        entry["lexicalisations"] = {
            lang_code: [{"comment": "", "lex": str(sample.get("input_text", "")).strip()}]
        }
        entries.append(entry)
    return {"entries": entries}


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_entry_key(entry: Dict[str, Any], idx: int) -> str:
    return str(entry.get("xml_id") or entry.get("id_short") or entry.get("sample_id") or f"idx_{idx}")


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


def set_entry_text(entry: Dict[str, Any], lang_code: str, text: str) -> None:
    entry["lexicalisations"] = {lang_code: [{"comment": "", "lex": text}]}


def triples_to_strings(entry: Dict[str, Any]) -> List[str]:
    triples = []
    for triple in entry.get("modifiedtripleset", []):
        triples.append(f"{triple.get('subject', '')}|{triple.get('property', '')}|{triple.get('object', '')}")
    return triples


def format_triple_list(items: Iterable[str]) -> str:
    items = [str(item).strip() for item in items if str(item).strip()]
    if not items:
        return "- None"
    return "\n".join(f"- {item}" for item in items)


def parse_triple_fields(triple_str: str) -> Optional[Dict[str, str]]:
    triple = str(triple_str).strip()
    if not triple:
        return None

    labeled = re.match(r"^\[S\]\s*(.*?)\s*\[P\]\s*(.*?)\s*\[O\]\s*(.*)$", triple)
    if labeled:
        return {
            "subject": labeled.group(1).strip(),
            "predicate": labeled.group(2).strip(),
            "object": labeled.group(3).strip(),
        }

    parts = [part.strip() for part in triple.split("|")]
    if len(parts) == 3:
        return {"subject": parts[0], "predicate": parts[1], "object": parts[2]}
    return None


def format_triple_for_repair(triple_str: str, fmt: str) -> str:
    triple = str(triple_str).strip()
    fields = parse_triple_fields(triple)

    if fmt == "json":
        if fields is None:
            return json.dumps({"raw": triple}, ensure_ascii=False)
        return json.dumps(fields, ensure_ascii=False)

    if fields is None:
        return evaluator.format_triple(triple)
    return f"[S] {fields['subject']} [P] {fields['predicate']} [O] {fields['object']}"


def format_triple_block(items: Iterable[str], fmt: str) -> str:
    normalized = [str(item).strip() for item in items if str(item).strip()]
    if fmt == "json":
        rows = []
        for item in normalized:
            fields = parse_triple_fields(item)
            rows.append(fields if fields is not None else {"raw": item})
        return json.dumps(rows, indent=2, ensure_ascii=False)
    return "\n".join(f"- {format_triple_for_repair(item, fmt)}" for item in normalized)


def format_source_triples_for_prompt(items: Iterable[str], fmt: str) -> str:
    normalized = [str(item).strip() for item in items if str(item).strip()]
    if not normalized:
        return "[]"
    return format_triple_block(normalized, fmt)


def render_feedback_section(title: str, items: List[str], guidance: str, fmt: str) -> str:
    normalized = [str(item).strip() for item in items if str(item).strip()]
    if not normalized:
        return ""
    return f"""{title}:
{format_triple_block(normalized, fmt)}
Guidance:
{guidance}

"""


def simplify_descriptions(descriptions: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    simplified = copy.deepcopy(descriptions or {})
    for lang, props in simplified.items():
        for prop, text in props.items():
            text = re.sub(r"\(.*?\)", "", str(text))
            simplified[lang][prop] = re.split(r"[.;:]", text)[0].strip()
    return simplified


def build_repair_user_payload(args: argparse.Namespace, entry: Dict[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    descriptions = simplify_descriptions(entry.get("descriptions", {}))

    if args.descriptions in {"all", "properties"}:
        merged_descriptions = descriptions.get("en", {}).copy()
        if args.target_lang != "en":
            target_descriptions = descriptions.get(args.target_lang, {})
            for key, value in target_descriptions.items():
                if key in merged_descriptions:
                    merged_descriptions[key] += f". {value}"
                else:
                    merged_descriptions[key] = value
        if args.descriptions == "all":
            payload["descriptions"] = merged_descriptions
        else:
            entry_props = {triple.get("property", "") for triple in entry.get("modifiedtripleset", [])}
            payload["property-descriptions"] = {
                key: value for key, value in merged_descriptions.items() if key in entry_props
            }

    if args.labels:
        labels = entry.get("labels", {}).get(args.target_lang, {})
        payload["labels"] = json.dumps(labels, indent=4, ensure_ascii=False)

    payload["data"] = [
        {
            "id": idx,
            "subject": triple.get("subject", ""),
            "property": triple.get("property", ""),
            "object": triple.get("object", ""),
        }
        for idx, triple in enumerate(entry.get("modifiedtripleset", []))
    ]
    return payload


def is_perfect(parsed_output: Dict[str, List[str]]) -> bool:
    return not any(parsed_output.get(name) for name in ("missing", "extra", "incorrect"))


def clear_cuda_memory() -> None:
    """Aggressively clean up between vLLM engine instances.

    Even with VLLM_USE_V1=0, simply `del`-ing an LLM does not always release
    worker subprocesses + /dev/shm artifacts immediately. This belt-and-suspenders
    cleanup avoids `RuntimeError: cancelled` in the next engine's SHM broadcast
    by:
      1. Multiple GC passes to release Python-side references.
      2. Clear CUDA caches + IPC handles, then synchronize.
      3. Sleep so vLLM worker subprocesses can exit naturally.
      4. Manually remove leftover /dev/shm/{vllm,psm}_* owned by current user.
    """
    import glob
    import time

    # 1. Force-release Python references so dangling vLLM objects can finalize.
    for _ in range(3):
        gc.collect()

    # 2. CUDA-side cleanup (best-effort).
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
    except Exception:
        pass

    # 3. Give vLLM worker subprocesses time to exit gracefully.
    time.sleep(5)

    # 4. Remove leftover SHM files from previous engine instances. We only
    #    touch files owned by the current user to stay safe on shared hosts.
    try:
        my_uid = os.getuid()
    except AttributeError:
        my_uid = None
    for pattern in ("/dev/shm/vllm_*", "/dev/shm/psm_*"):
        for shm_file in glob.glob(pattern):
            try:
                if my_uid is None or os.stat(shm_file).st_uid == my_uid:
                    os.remove(shm_file)
            except OSError:
                pass

    gc.collect()
    time.sleep(1)


def make_eval_args(args: argparse.Namespace, seed: int) -> SimpleNamespace:
    return SimpleNamespace(
        model_cache=args.model_cache,
        dtype=args.evaluator_dtype,
        tensor_parallel_size=args.evaluator_tensor_parallel_size,
        gpu_memory_utilization=args.evaluator_gpu_memory_utilization,
        max_model_len=args.evaluator_max_model_len,
        max_num_seqs=args.evaluator_max_num_seqs,
        quantization=args.evaluator_quantization,
        enforce_eager=args.evaluator_enforce_eager,
        max_lora_rank=args.evaluator_max_lora_rank,
        use_hf=args.evaluator_use_hf,
        max_tokens=args.evaluator_max_tokens,
        top_p=args.evaluator_top_p,
        top_k=args.evaluator_top_k,
        repetition_penalty=args.evaluator_repetition_penalty,
        seed=seed,
        batch_size=args.evaluator_batch_size,
    )


def run_judge_round(
    args: argparse.Namespace,
    data: Dict[str, Any],
    current_texts: Dict[int, str],
    active_indices: List[int],
    round_idx: int,
    output_dir: Path,
) -> Tuple[List[Dict[str, Any]], List[int], List[int]]:
    if not active_indices:
        return [], [], []

    # Local resume: if round{N}_eval_active.json already covers all active_indices,
    # skip the vLLM judge and reuse it. Force re-judge by deleting the file.
    local_eval_path = output_dir / f"round{round_idx}_eval_active.json"
    if local_eval_path.exists() and local_eval_path.stat().st_size > 1000:
        try:
            cached_rows = load_json(local_eval_path)
            if isinstance(cached_rows, list):
                by_index: Dict[int, Dict[str, Any]] = {}
                for row in cached_rows:
                    if isinstance(row, dict) and "index" in row and "parsed_output" in row:
                        by_index[int(row["index"])] = row
                if all(int(idx) in by_index for idx in active_indices):
                    print(
                        f"[resume] round {round_idx}: reusing local judge output "
                        f"({len(cached_rows)} rows from {local_eval_path.name})"
                    )
                    results = []
                    for idx in active_indices:
                        row = copy.deepcopy(by_index[int(idx)])
                        row["round"] = round_idx
                        if "perfect" not in row:
                            row["perfect"] = is_perfect(row.get("parsed_output", {}))
                        results.append(row)
                    perfect_indices = [int(r["index"]) for r in results if r["perfect"]]
                    imperfect_indices = [int(r["index"]) for r in results if not r["perfect"]]
                    return results, perfect_indices, imperfect_indices
                else:
                    missing_count = sum(1 for idx in active_indices if int(idx) not in by_index)
                    print(
                        f"[resume] round {round_idx}: local eval covers only "
                        f"{len(by_index)}/{len(active_indices)} active indices "
                        f"({missing_count} missing), will re-judge"
                    )
        except Exception as exc:
            print(f"[resume] failed to load {local_eval_path.name}: {exc}, will re-judge")

    if round_idx == 1 and args.reuse_round1_eval_json:
        reuse_path = ensure_file(Path(args.reuse_round1_eval_json))
        reused_rows = load_json(reuse_path)
        if not isinstance(reused_rows, list):
            raise ValueError(f"Expected a JSON list in {reuse_path}")
        by_index: Dict[int, Dict[str, Any]] = {}
        for row_idx, row in enumerate(reused_rows):
            if not isinstance(row, dict) or "parsed_output" not in row:
                raise ValueError(f"Malformed reused eval row in {reuse_path}")
            row_copy = copy.deepcopy(row)
            row_copy["index"] = int(row_copy.get("index", row_idx))
            row_copy["round"] = round_idx
            row_copy["perfect"] = bool(row_copy.get("perfect", is_perfect(row_copy["parsed_output"])))
            by_index[int(row_copy["index"])] = row_copy
        missing = [idx for idx in active_indices if idx not in by_index]
        if missing:
            raise ValueError(
                f"Reused round1 eval file does not cover all active indices. Missing: {missing[:10]}"
            )
        results = [by_index[idx] for idx in active_indices]
        perfect_indices = [row["index"] for row in results if row["perfect"]]
        imperfect_indices = [row["index"] for row in results if not row["perfect"]]
        save_json(output_dir / f"round{round_idx}_eval_active.json", results)
        save_json(
            output_dir / f"round{round_idx}_eval_summary.json",
            {
                "round": round_idx,
                "judged": len(results),
                "perfect": len(perfect_indices),
                "imperfect": len(imperfect_indices),
                "temperature": args.judge_temperature,
                "seed": args.judge_seed,
                "evaluator_model": args.evaluator_model,
                "reused_eval_json": str(reuse_path),
                "summary": evaluator.summarize_results(results) if results else {},
            },
        )
        return results, perfect_indices, imperfect_indices

    model_key = args.evaluator_model
    config = evaluator.MODEL_CONFIGS[model_key]
    eval_args = make_eval_args(args, args.judge_seed)
    eval_args.temperature = args.judge_temperature

    evaluator.set_seed(args.judge_seed)
    engine = None
    template = None
    previous_use_hf = None
    results: List[Dict[str, Any]] = []

    try:
        engine, template, previous_use_hf = evaluator.load_vllm_engine(eval_args, config)
        infer_request_cls = evaluator.import_swift_api()["InferRequest"]
        request_config = evaluator.build_request_config(eval_args, args.judge_temperature)
        batch_size = int(args.evaluator_batch_size or config["batch_size"])

        for start in tqdm(range(0, len(active_indices), batch_size), desc=f"judge r{round_idx}"):
            batch_indices = active_indices[start : start + batch_size]
            infer_requests = []
            prompts = []
            for idx in batch_indices:
                entry = data["entries"][idx]
                source_triples = triples_to_strings(entry)
                query = evaluator.build_query(current_texts[idx], source_triples)
                prompts.append(query)
                infer_requests.append(
                    evaluator.make_infer_request(
                        infer_request_cls,
                        query,
                        config.get("default_system"),
                        pass_system=template is None,
                    )
                )

            infer_kwargs = {"use_tqdm": False}
            if template is not None:
                infer_kwargs["template"] = template
            responses = engine.infer(infer_requests, request_config, **infer_kwargs)

            for offset, (idx, prompt, resp) in enumerate(zip(batch_indices, prompts, responses)):
                entry = data["entries"][idx]
                response = resp.choices[0].message.content
                parsed = evaluator.parse_prediction(response)
                source_triples = triples_to_strings(entry)
                metrics = evaluator.compute_prf(parsed, len(source_triples))
                results.append(
                    {
                        "sample_id": get_entry_key(entry, idx),
                        "index": idx,
                        "round": round_idx,
                        "dataset_name": args.dataset_name,
                        "baseline_name": args.baseline_name,
                        "input_text": current_texts[idx],
                        "source_triples": source_triples,
                        "raw_source_triples": entry.get("modifiedtripleset", []),
                        "prompt": prompt,
                        "model_output": response,
                        "parsed_output": parsed,
                        "metrics": metrics,
                        "perfect": is_perfect(parsed),
                    }
                )

    finally:
        evaluator.restore_use_hf(previous_use_hf)
        del engine, template
        clear_cuda_memory()

    perfect_indices = [row["index"] for row in results if row["perfect"]]
    imperfect_indices = [row["index"] for row in results if not row["perfect"]]

    save_json(output_dir / f"round{round_idx}_eval_active.json", results)
    save_json(
        output_dir / f"round{round_idx}_eval_summary.json",
        {
            "round": round_idx,
            "judged": len(results),
            "perfect": len(perfect_indices),
            "imperfect": len(imperfect_indices),
            "temperature": args.judge_temperature,
            "seed": args.judge_seed,
            "evaluator_model": model_key,
            "summary": evaluator.summarize_results(results) if results else {},
        },
    )
    return results, perfect_indices, imperfect_indices


def save_cumulative_eval(
    output_dir: Path,
    round_idx: int,
    all_eval_by_idx: Dict[int, Dict[str, Any]],
) -> None:
    rows = [all_eval_by_idx[idx] for idx in sorted(all_eval_by_idx)]
    save_json(output_dir / f"round{round_idx}_eval.json", rows)
    save_json(
        output_dir / f"round{round_idx}_eval_cumulative_summary.json",
        {
            "round": round_idx,
            "num_samples": len(rows),
            "perfect": sum(1 for row in rows if row.get("perfect")),
            "imperfect": sum(1 for row in rows if not row.get("perfect")),
            "summary": evaluator.summarize_results(rows) if rows else {},
        },
    )


def build_repair_instruction(
    args: argparse.Namespace,
    entry: Dict[str, Any],
    parsed_output: Dict[str, List[str]],
    source_triples: List[str],
    previous_text: str,
) -> str:
    variant = get_prompt_variant(getattr(args, "repair_prompt_variant", DEFAULT_REPAIR_PROMPT_VARIANT))
    repair_format = getattr(args, "repair_feedback_format", "spo")
    source_context = format_source_triples_for_prompt(source_triples, repair_format)
    previous_text_block = json.dumps({"previous_text": previous_text}, indent=2, ensure_ascii=False)
    source_payload = build_repair_user_payload(args, entry)
    source_payload_block = ""
    if any(key != "data" for key in source_payload):
        rendered_payload = json.dumps(source_payload, indent=4, ensure_ascii=False).replace("_", " ")
        source_payload_block = f"""Current sample:
{rendered_payload}

"""
    task_block = "\n".join(variant["task_lines"])
    revision_lines = "\n".join(f"- {line}" for line in variant["revision_requirements"])

    return f"""You are revising a previously generated data-to-text output.

{source_payload_block}\
# Source triples
{source_context}

# Task
{task_block}

# Previous text
{previous_text_block}

# Verifier feedback

{render_feedback_section(
    "Missing facts to express",
    parsed_output.get("missing", []),
    variant["missing_guidance"],
    repair_format,
)}{render_feedback_section(
    "Unsupported facts to remove",
    parsed_output.get("extra", []),
    variant["extra_guidance"],
    repair_format,
)}{render_feedback_section(
    "Potentially inconsistent facts to check",
    parsed_output.get("incorrect", []),
    variant["incorrect_guidance"],
    repair_format,
)}Revision requirements:
{revision_lines}
"""


def get_lexicalisation(entry: Dict[str, Any], lang_code: str) -> str:
    lexicalisations = entry.get("lexicalisations", {})
    lang_lexes = lexicalisations.get(lang_code) or lexicalisations.get("en") or []
    if not lang_lexes:
        return ""
    first = lang_lexes[0]
    return str(first.get("lex") or first.get("comment") or "").strip()


def build_generation_user_content(args: argparse.Namespace, entry: Dict[str, Any], language: str) -> str:
    payload = build_repair_user_payload(args, entry)
    payload_text = json.dumps(payload, indent=4, ensure_ascii=False).replace("_", " ")
    num_triples = len(entry.get("modifiedtripleset", []))
    return (
        payload_text
        + "\n\n"
        + f"Generate a lexicalisation of all the {num_triples} triples in {language}.\n"
        + "When necessary, generate text from other languages in their original script.\n"
        + "Include all the information from the triples regardless of their type or relevance."
    )


def select_lightweight_fewshots(
    args: argparse.Namespace,
    target_entry: Dict[str, Any],
    fewshot_entries: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Select 4-shot D2T examples without loading LaBSE during repair.

    This mirrors the original overlap intuition while avoiding a SentenceTransformer
    model competing with vLLM for GPU memory in the repair process.
    """

    target_props = [triple.get("property", "") for triple in target_entry.get("modifiedtripleset", [])]
    target_prop_set = set(target_props)
    target_size = len(target_props)

    candidates = []
    for idx, entry in enumerate(fewshot_entries):
        lex = get_lexicalisation(entry, args.fewshot_lang)
        if not lex:
            continue
        props = [triple.get("property", "") for triple in entry.get("modifiedtripleset", [])]
        prop_set = set(props)
        overlap = len(target_prop_set & prop_set)
        candidates.append((idx, entry, overlap, len(props)))

    candidates.sort(key=lambda item: (-item[2], abs(item[3] - target_size), item[3], item[0]))
    return [entry for _, entry, _, _ in candidates[: args.num_fewshot]]


def build_labse_fewshot_state(args: argparse.Namespace, fewshot_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    from sentence_transformers import SentenceTransformer
    import faiss

    all_props = sorted(
        {
            triple.get("property", "")
            for entry in fewshot_entries
            for triple in entry.get("modifiedtripleset", [])
            if triple.get("property", "")
        }
    )
    model = SentenceTransformer("LaBSE", device="cpu")
    embeddings = model.encode(all_props, show_progress_bar=False)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return {
        "mode": "labse_cpu",
        "entries": fewshot_entries,
        "model": model,
        "all_props": all_props,
        "all_props_set": set(all_props),
        "index": index,
    }


def build_history_fewshot_state(args: argparse.Namespace, fewshot_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    if args.repair_history_fewshot_selector == "labse_cpu" and args.fewshot_style == "overlap":
        return build_labse_fewshot_state(args, fewshot_entries)
    return {"mode": "lightweight", "entries": fewshot_entries}


def get_labse_similar_props(search_properties: List[str], state: Dict[str, Any]) -> List[str]:
    if not search_properties:
        return []
    embeddings = state["model"].encode(search_properties, show_progress_bar=False)
    _, indices = state["index"].search(embeddings, 1)
    return [state["all_props"][idx] for idx in indices[:, 0]]


def select_labse_fewshots(
    args: argparse.Namespace,
    target_entry: Dict[str, Any],
    state: Dict[str, Any],
) -> List[Dict[str, Any]]:
    target_props = [triple.get("property", "") for triple in target_entry.get("modifiedtripleset", [])]
    seen_properties = [prop for prop in target_props if prop in state["all_props_set"]]
    unseen_properties = [prop for prop in target_props if prop not in state["all_props_set"]]
    similar_properties = get_labse_similar_props(unseen_properties, state)

    original_search_properties = list(set(seen_properties + similar_properties))
    original_size = len(target_props)

    max_size = max((len(entry.get("modifiedtripleset", [])) for entry in state["entries"]), default=original_size)
    size_order = [original_size]
    for offset in range(1, max_size - original_size + 1):
        size_order.append(original_size + offset)
    for offset in range(1, original_size):
        size_order.append(original_size - offset)
    size_rank = {size: rank for rank, size in enumerate(size_order)}

    metas = []
    for idx, entry in enumerate(state["entries"]):
        lex = get_lexicalisation(entry, args.fewshot_lang)
        if not lex:
            continue
        props = [triple.get("property", "") for triple in entry.get("modifiedtripleset", [])]
        metas.append({"idx": idx, "entry": entry, "props": props, "props_set": set(props), "size": len(props)})

    selected: List[Dict[str, Any]] = []
    selected_ids = set()
    search_properties = original_search_properties[:]

    while len(selected) < args.num_fewshot and len(selected_ids) < len(metas):
        search_set = set(search_properties)
        sorted_metas = sorted(
            metas,
            key=lambda item: (
                -len(item["props_set"] & search_set),
                size_rank.get(item["size"], float("inf")),
                item["idx"],
            ),
        )
        chosen = None
        for item in sorted_metas:
            if item["idx"] not in selected_ids:
                chosen = item
                break
        if chosen is None:
            break

        selected.append(chosen["entry"])
        selected_ids.add(chosen["idx"])

        if set(search_properties) - chosen["props_set"] == set(search_properties):
            search_properties = original_search_properties[:]
        search_properties = list(set(search_properties) - chosen["props_set"])
        if not search_properties:
            search_properties = original_search_properties[:]

    return selected


def select_history_fewshots(
    args: argparse.Namespace,
    target_entry: Dict[str, Any],
    fewshot_state: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if fewshot_state.get("mode") == "labse_cpu":
        return select_labse_fewshots(args, target_entry, fewshot_state)
    return select_lightweight_fewshots(args, target_entry, fewshot_state["entries"])


def build_d2t_history_messages(
    args: argparse.Namespace,
    entry: Dict[str, Any],
    previous_text: str,
    fewshot_state: Dict[str, Any],
) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    for fewshot in select_history_fewshots(args, entry, fewshot_state):
        messages.append(
            {
                "role": "user",
                "content": build_generation_user_content(args, fewshot, args.fewshot_language),
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": json.dumps(
                    {"full-text": get_lexicalisation(fewshot, args.fewshot_lang)},
                    indent=4,
                    ensure_ascii=False,
                ),
            }
        )

    messages.append(
        {
            "role": "user",
            "content": build_generation_user_content(args, entry, args.target_language),
        }
    )
    messages.append(
        {
            "role": "assistant",
            "content": json.dumps({"full-text": previous_text}, indent=4, ensure_ascii=False),
        }
    )
    return messages


def make_repair_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        input_file=str(args.baseline_json),
        fewshot_source=str(args.fewshot_source) if args.fewshot_source else None,
        output_dir=str(args.output_dir),
        model_cache=args.model_cache,
        hf_token="",
        model_source=args.generator_model_source,
        model_id_or_path=args.generator_model_id_or_path,
        model_size=args.generator_model_size,
        model_type=args.generator_model_type,
        target_lang=args.target_lang,
        output_filename="",
        lang_code=args.lang_code,
        fewshot_lang=args.fewshot_lang,
        fewshot_style=args.fewshot_style,
        generation_technique=args.generation_technique,
        descriptions=args.descriptions,
        labels=args.labels,
        num_fewshot=args.num_fewshot,
        do_print_check=False,
        disable_thinking=args.disable_thinking,
        debug_prompt_tokens=False,
        debug_prompt_token_count=16,
        debug_prompt_char_count=240,
        dtype=args.generator_dtype,
        tensor_parallel_size=args.generator_tensor_parallel_size,
        gpu_memory_utilization=args.generator_gpu_memory_utilization,
        max_model_len=args.generator_max_model_len,
        quantization=args.generator_quantization,
        trust_remote_code=args.generator_trust_remote_code,
        enforce_eager=args.generator_enforce_eager,
        max_num_seqs=args.generator_max_num_seqs,
        batch_size=args.repair_batch_size,
        seed=args.repair_seed,
        max_new_tokens=args.repair_max_new_tokens,
        temperature=args.repair_temperature,
        top_p=args.repair_top_p,
        top_k=args.repair_top_k,
        repetition_penalty=args.repair_repetition_penalty,
        target_language=args.target_language,
        fewshot_language=args.fewshot_language,
        repair_feedback_format=args.repair_feedback_format,
        repair_context_mode=args.repair_context_mode,
        repair_history_fewshot_selector=args.repair_history_fewshot_selector,
    )


def parse_prediction_result(raw_text: str) -> Tuple[str, str]:
    from generate_vllm_simple import parse_prediction

    try:
        parsed_json = json.loads(raw_text)
        if isinstance(parsed_json, dict):
            return str(parsed_json.get("full-text", "")).strip(), "json"
        return str(parsed_json).strip(), "json_non_dict"
    except Exception:
        pass

    parsed = parse_prediction(raw_text)
    if isinstance(parsed, dict):
        text = str(parsed.get("full-text", "")).strip()
        if text.startswith("ERROR "):
            return text[len("ERROR ") :].strip(), "error_prefix"
        if "full-text" in raw_text:
            return text, "regex"
        return text, "dict_fallback"
    return str(parsed).strip(), "string_fallback"


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

    repair_args = make_repair_args(args)
    fewshot_state: Dict[str, Any] = {}
    if args.repair_context_mode == "d2t_history":
        fewshot_payload = load_json(args.fewshot_source)
        if not isinstance(fewshot_payload, dict) or not isinstance(fewshot_payload.get("entries"), list):
            raise ValueError(f"Expected fewshot_source with an 'entries' list: {args.fewshot_source}")
        fewshot_state = build_history_fewshot_state(repair_args, fewshot_payload["entries"])

    os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
    if repair_args.model_type == "qwen3" and os.environ.get("USE_HF") is None:
        os.environ["USE_HF"] = "1"

    from generate_vllm_simple import configure_multiprocessing, set_seed

    configure_multiprocessing()
    set_seed(args.repair_seed)

    llm = None
    repaired_texts: Dict[int, str] = {}
    repair_records: List[Dict[str, Any]] = []

    try:
        llm = load_vllm(repair_args, args.generator_model_id_or_path)
        tokenizer = llm.get_tokenizer()
        sampling_params = build_sampling_params(repair_args, tokenizer)
        prompt_variant = get_prompt_variant(
            getattr(args, "repair_prompt_variant", DEFAULT_REPAIR_PROMPT_VARIANT)
        )

        pending = []
        for idx in tqdm(repair_indices, desc=f"repair r{round_idx} build"):
            entry = data["entries"][idx]
            messages = [{"role": "system", "content": prompt_variant["repair_system"]}]
            if args.repair_context_mode == "d2t_history":
                messages.extend(build_d2t_history_messages(repair_args, entry, current_texts[idx], fewshot_state))
            messages.append(
                {
                    "role": "user",
                    "content": build_repair_instruction(
                        repair_args,
                        entry,
                        eval_by_idx[idx]["parsed_output"],
                        triples_to_strings(entry),
                        current_texts[idx],
                    ),
                }
            )
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
                        "feedback": eval_by_idx[item["idx"]]["parsed_output"],
                        "messages": item["messages"],
                    }
                )

    finally:
        del llm
        clear_cuda_memory()

    save_json(output_dir / f"round{round_idx}_repair_records.json", repair_records)
    return repaired_texts


def save_generation_snapshot(
    args: argparse.Namespace,
    data: Dict[str, Any],
    current_texts: Dict[int, str],
    output_dir: Path,
    round_idx: int,
) -> Path:
    snapshot = copy.deepcopy(data)
    for idx, entry in enumerate(snapshot["entries"]):
        set_entry_text(entry, args.lang_code, current_texts[idx])
    path = output_dir / f"round{round_idx}_improved.json"
    save_json(path, snapshot)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Iteratively repair D2T outputs with vLLM + LoRA evaluator feedback.")

    parser.add_argument("--baseline_json", type=Path, required=True)
    parser.add_argument("--fewshot_source", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_rounds", type=int, default=3)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--dataset_name", type=str, default="")
    parser.add_argument("--baseline_name", type=str, default="")

    parser.add_argument("--target_lang", type=str, default="en")
    parser.add_argument("--lang_code", type=str, default="en")
    parser.add_argument("--fewshot_lang", type=str, default="en")
    parser.add_argument("--fewshot_style", type=str, default="overlap")
    parser.add_argument("--generation_technique", type=str, default="dir")
    parser.add_argument("--descriptions", type=str, default="none")
    parser.add_argument("--labels", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=4)

    parser.add_argument("--evaluator_model", choices=sorted(evaluator.MODEL_CONFIGS), required=True)
    parser.add_argument("--judge_temperature", type=float, default=0.3)
    parser.add_argument("--judge_seed", type=int, default=2023)
    parser.add_argument("--evaluator_batch_size", type=int, default=1)
    parser.add_argument("--evaluator_max_tokens", type=int, default=1024)
    parser.add_argument("--evaluator_dtype", type=str, default="bfloat16")
    parser.add_argument("--evaluator_tensor_parallel_size", type=int, default=1)
    parser.add_argument("--evaluator_gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--evaluator_max_model_len", type=int, default=4096)
    parser.add_argument("--evaluator_max_num_seqs", type=int, default=16)
    parser.add_argument("--evaluator_quantization", type=str, default="")
    parser.add_argument("--evaluator_enforce_eager", action="store_true")
    parser.add_argument("--evaluator_max_lora_rank", type=int, default=16)
    parser.add_argument("--evaluator_use_hf", choices=["auto", "0", "1"], default="1")
    parser.add_argument("--evaluator_top_p", type=float, default=None)
    parser.add_argument("--evaluator_top_k", type=int, default=None)
    parser.add_argument("--evaluator_repetition_penalty", type=float, default=None)
    parser.add_argument("--reuse_round1_eval_json", type=Path, default=None)

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
    parser.add_argument("--repair_feedback_format", choices=["spo", "json"], default="spo")
    parser.add_argument("--repair_context_mode", choices=["single", "d2t_history"], default="single")
    parser.add_argument("--repair_history_fewshot_selector", choices=["lightweight", "labse_cpu"], default="lightweight")
    parser.add_argument("--repair_prompt_variant", choices=REPAIR_PROMPT_VARIANTS, default=DEFAULT_REPAIR_PROMPT_VARIANT)
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


def main() -> None:
    args = parse_args()
    os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    args.baseline_json = ensure_file(args.baseline_json)
    if args.repair_context_mode == "d2t_history":
        if args.fewshot_source is None:
            raise ValueError("--fewshot_source is required when --repair_context_mode=d2t_history")
        args.fewshot_source = ensure_file(args.fewshot_source)
    else:
        args.fewshot_source = None
    output_dir = ensure_dir(args.output_dir)

    language_dict = {
        "br": "Breton",
        "cy": "Welsh",
        "en": "English",
        "fr": "French",
        "ga": "Irish",
        "it": "Italian",
        "mt": "Maltese",
        "pt": "Portuguese",
        "ru": "Russian",
        "es": "Spanish",
        "zh": "Simplified Chinese",
    }
    args.target_language = language_dict[args.target_lang]
    args.fewshot_language = language_dict[args.fewshot_lang]

    data = normalize_generation_payload(load_json(args.baseline_json), args.lang_code)
    if args.max_samples is not None:
        data = copy.deepcopy(data)
        data["entries"] = data["entries"][: args.max_samples]

    current_texts = {idx: get_text_from_entry(entry, args.lang_code) for idx, entry in enumerate(data["entries"])}
    active_indices = list(range(len(data["entries"])))
    all_eval_by_idx: Dict[int, Dict[str, Any]] = {}

    save_json(output_dir / "round0_baseline.json", data)
    save_generation_snapshot(args, data, current_texts, output_dir, 0)

    run_summary = {
        "baseline_json": str(args.baseline_json),
        "output_dir": str(output_dir),
        "evaluator_model": args.evaluator_model,
        "reuse_round1_eval_json": str(args.reuse_round1_eval_json) if args.reuse_round1_eval_json else None,
        "judge_temperature": args.judge_temperature,
        "judge_seed": args.judge_seed,
        "generator_model_id_or_path": args.generator_model_id_or_path,
        "repair_temperature": args.repair_temperature,
        "repair_seed": args.repair_seed,
        "repair_feedback_format": args.repair_feedback_format,
        "repair_context_mode": args.repair_context_mode,
        "repair_history_fewshot_selector": (
            args.repair_history_fewshot_selector if args.repair_context_mode == "d2t_history" else None
        ),
        "repair_prompt_variant": args.repair_prompt_variant,
        "rounds": [],
    }

    for round_idx in range(1, args.max_rounds + 1):
        # Round-level resume: if round{N}_improved.json + round{N}_eval.json both exist,
        # the round is fully complete -- reload state and continue to next round.
        snapshot_path = output_dir / f"round{round_idx}_improved.json"
        cum_eval_path = output_dir / f"round{round_idx}_eval.json"
        if (snapshot_path.exists()
                and snapshot_path.stat().st_size > 10000
                and cum_eval_path.exists()
                and cum_eval_path.stat().st_size > 1000):
            try:
                snap_data = load_json(snapshot_path)
                cum_rows = load_json(cum_eval_path)
                if (isinstance(snap_data, dict) and "entries" in snap_data
                        and isinstance(cum_rows, list)
                        and len(snap_data["entries"]) >= len(data["entries"])):
                    for idx in range(len(data["entries"])):
                        text = get_text_from_entry(snap_data["entries"][idx], args.lang_code)
                        if text:
                            current_texts[idx] = text
                    for row in cum_rows:
                        if isinstance(row, dict) and "index" in row:
                            all_eval_by_idx[int(row["index"])] = row
                    next_imperfect = [int(row["index"]) for row in cum_rows if not row.get("perfect")]
                    print(
                        f"[resume] round {round_idx}: skipped (loaded snapshot + "
                        f"{len(cum_rows)} eval rows, {len(next_imperfect)} imperfect remaining)"
                    )
                    run_summary["rounds"].append({
                        "round": round_idx,
                        "judged": len(cum_rows),
                        "new_perfect": sum(1 for r in cum_rows if r.get("perfect")),
                        "remaining_imperfect": len(next_imperfect),
                        "resumed": True,
                    })
                    if not next_imperfect:
                        break
                    active_indices = next_imperfect
                    continue
            except Exception as exc:
                print(f"[resume] failed to load round {round_idx} state: {exc}, re-running")

        eval_rows, perfect_indices, imperfect_indices = run_judge_round(
            args=args,
            data=data,
            current_texts=current_texts,
            active_indices=active_indices,
            round_idx=round_idx,
            output_dir=output_dir,
        )
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
