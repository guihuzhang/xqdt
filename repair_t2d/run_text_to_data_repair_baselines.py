#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import gc
import json
import math
import os
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
OUTPUT_ROOT = SCRIPT_DIR / "t2d_repair_output"
DEFAULT_BASELINE_OUTPUT_ROOT = OUTPUT_ROOT / "baseline" / "round1"
for import_root in (SCRIPT_DIR, REPO_ROOT):
    import_root_str = str(import_root)
    if import_root_str not in sys.path:
        sys.path.insert(0, import_root_str)

import run_text_to_data_verifier_repair as core  # noqa: E402


SUPPORTED_BASELINE_MODES = (
    "pive_missing",
    "self_refine_v2",
    "nli_binary",
    "factspotter_binary",
)

DEFAULT_NLI_TEMPLATE_PATH = SCRIPT_DIR / "webnlg_templates.json"

FACTSPOTTER_HF_MODELS = {
    "deberta_small": "Inria-CEDAR/FactSpotter-DeBERTaV3-Small",
    "deberta_base": "Inria-CEDAR/FactSpotter-DeBERTaV3-Base",
    "deberta_large": "Inria-CEDAR/FactSpotter-DeBERTaV3-Large",
}

PIVE_VERIFIER_MODEL_ID = "jiuzhouh/flan-t5-xxl-lora-verifier"
PIVE_REPAIR_EXAMPLE = (
    "Transform the text into a semantic graph and also add the given triples to the generated semantic graph.\n"
    "Example:\n"
    "Text: Sportpark De Toekomst is located in Ouder-Amstel, Netherlands. It is owned and operated by AFC Ajax N.V. and their tenants include the Ajax Youth Academy. \n"
    "Triples: [\"Sportpark De Toekomst\", \"country\", \"Netherlands\"], "
    "[\"Sportpark De Toekomst\", \"operator\", \"AFC Ajax N.V.\"]\n"
    "Semantic graph: [[\"Sportpark De Toekomst\", \"location\", \"Ouder-Amstel\"], "
    "[\"Sportpark De Toekomst\", \"country\", \"Netherlands\"], "
    "[\"Sportpark De Toekomst\", \"owner\", \"AFC Ajax N.V.\"], "
    "[\"Sportpark De Toekomst\", \"operator\", \"AFC Ajax N.V.\"], "
    "[\"Sportpark De Toekomst\", \"tenant\", \"Ajax Youth Academy\"]]\n"
)

SELF_REFINE_FEEDBACK_SYSTEM = "You are a careful text-to-data critic."
SELF_REFINE_REPAIR_SYSTEM = "You are a careful text-to-data repair assistant."

NLI_TEMPLATE_REMAP = {
    "eat_type": "eatType",
    "rating": "customer rating",
    "family_friendly": "familyFriendly",
    "price_range": "priceRange",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pure baseline text-to-data repair methods without changing the verifier-guided main path.")
    parser.add_argument("--datasets", nargs="+", default=["genwiki", "webnlg"], choices=sorted(core.UNIFIED_DATASETS))
    parser.add_argument("--baselines", nargs="+", default=["bt5", "regen"], choices=["bt5", "regen", "fewshot"])
    parser.add_argument("--repair-models", nargs="+", default=["gemma3_4b", "gemma3_12b", "gemma3_27b", "llama3_1_8b", "llama3_2_3b", "llama3_3_70b_awq", "qwen3_1.7b", "qwen3_4b", "qwen3_8b", "qwen3_14b", "qwen3_32b"], choices=sorted(core.REPAIR_MODEL_CONFIGS))
    parser.add_argument("--modes", nargs="+", default=["pive_missing"], choices=SUPPORTED_BASELINE_MODES)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--input-json", type=Path, default=None, help="Optional previous repair output json to use as the current triples input.")
    parser.add_argument("--only-previously-flagged", action="store_true", help="When used with --input-json, continue only entries whose previous output had flagged=true.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_BASELINE_OUTPUT_ROOT)
    parser.add_argument("--model-cache", type=str, default=".cache_text_to_data_repair")
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
    parser.add_argument("--repair-tp-overrides", nargs="*", default=[])
    parser.add_argument("--nli-template-file", type=Path, default=DEFAULT_NLI_TEMPLATE_PATH)
    parser.add_argument("--nli-threshold", type=float, default=0.5)
    parser.add_argument("--nli-global-threshold", type=float, default=0.5)
    parser.add_argument("--pive-signal-batch-size", type=int, default=4)
    parser.add_argument("--nli-signal-batch-size", type=int, default=64)
    parser.add_argument("--factspotter-signal-batch-size", type=int, default=64)
    parser.add_argument("--self-refine-feedback-batch-size", type=int, default=8)
    parser.add_argument("--factspotter-backend", type=str, default="deberta_base", choices=("deberta_small", "deberta_base", "deberta_large"))
    parser.add_argument("--factspotter-threshold", type=float, default=0.5)
    parser.add_argument("--pive-max-new-tokens", type=int, default=256)
    parser.add_argument("--shot-k", type=int, default=None, help="Override shot count in output filenames/metadata. Defaults to input payload shot_k or 0.")
    args = parser.parse_args()
    args.repair_tp_override_map = core.parse_tp_overrides(args.repair_tp_overrides)
    return args


def active_iteration_samples(samples: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [sample for sample in samples if sample.get("iteration_active", True)]


def signal_lock_path(output_root: Path, mode: str, dataset: str, baseline: str, repair_model: str) -> Path:
    cache_path = signal_cache_path(output_root, mode, dataset, baseline, repair_model)
    return cache_path.parent / f"{cache_path.name}.lock"


def load_signal_cache(cache_path: Path) -> Optional[List[Dict[str, Any]]]:
    if not cache_path.exists():
        return None
    payload = core.load_json(cache_path)
    if isinstance(payload, list):
        return payload
    return None


@contextmanager
def exclusive_file_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def build_skipped_signal_row(sample: Dict[str, Any], **extra: Any) -> Dict[str, Any]:
    row = {
        "sample_id": sample["sample_id"],
        "flagged": False,
        "skipped_iteration": True,
    }
    row.update(extra)
    return row


def build_markdown_table(columns: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join("---" for _ in columns) + "|"
    if not rows:
        return header + "\n" + separator
    lines = [header, separator]
    for row in rows:
        escaped = [str(cell).replace("\n", " ").replace("|", "/").strip() for cell in row]
        lines.append("| " + " | ".join(escaped) + " |")
    return "\n".join(lines)


def extract_first_markdown_table(raw_output: str) -> List[str]:
    lines = [line.rstrip() for line in str(raw_output).splitlines()]
    blocks: List[List[str]] = []
    current: List[str] = []
    for line in lines:
        if line.strip().startswith("|") and line.strip().endswith("|"):
            current.append(line.strip())
        else:
            if current:
                blocks.append(current)
                current = []
    if current:
        blocks.append(current)
    return max(blocks, key=len) if blocks else []


def is_markdown_separator_row(parts: Sequence[str]) -> bool:
    if not parts:
        return False
    for part in parts:
        cleaned = re.sub(r"[\-\:\s]", "", part)
        if cleaned:
            return False
    return True


def parse_markdown_table(raw_output: str, expected_columns: Sequence[str]) -> List[Dict[str, str]]:
    table_lines = extract_first_markdown_table(raw_output)
    if len(table_lines) < 2:
        return []

    def split_row(line: str) -> List[str]:
        return [cell.strip() for cell in line.strip().strip("|").split("|")]

    header = [cell.lower() for cell in split_row(table_lines[0])]
    expected = [column.lower() for column in expected_columns]
    if header != expected:
        return []
    if not is_markdown_separator_row(split_row(table_lines[1])):
        return []

    rows: List[Dict[str, str]] = []
    for line in table_lines[2:]:
        parts = split_row(line)
        if len(parts) != len(expected_columns):
            continue
        rows.append({column: value for column, value in zip(expected_columns, parts)})
    return rows


def parse_optional_triples_markdown(raw_output: str) -> Tuple[List[str], bool]:
    text = str(raw_output).strip()
    if re.fullmatch(r"(?i)\s*none\s*\.?\s*", text):
        return [], True
    rows = parse_markdown_table(raw_output, ("subject", "predicate", "object"))
    triples: List[str] = []
    for row in rows:
        parsed = core.parse_pipe_triple(f'{row["subject"]} | {row["predicate"]} | {row["object"]}')
        if parsed is not None:
            triples.append(f"{parsed[0]} | {parsed[1]} | {parsed[2]}")
    return core.triples_to_strings(core.strings_to_tuples(triples)), bool(rows)


def baseline_output_filename(dataset: str, baseline: str, repair_model: str, mode: str, shot_k: int = 0) -> str:
    return core.repair_output_filename(dataset, baseline, repair_model, mode, shot_k)


def signal_cache_path(output_root: Path, mode: str, dataset: str, baseline: str, repair_model: str) -> Path:
    return output_root / "signal_cache" / mode / dataset / baseline / repair_model / "signals.json"


def signal_progress_jsonl_path(output_root: Path, mode: str, dataset: str, baseline: str, repair_model: str) -> Path:
    return output_root / "signal_cache" / mode / dataset / baseline / repair_model / "signals.jsonl"


def triples_to_pive_graph(triples: Sequence[str]) -> str:
    rows = [[subject, predicate, obj] for subject, predicate, obj in core.strings_to_tuples(triples)]
    return json.dumps(rows, ensure_ascii=False)


def triples_to_pive_inline(triples: Sequence[str]) -> str:
    rows = []
    for subject, predicate, obj in core.strings_to_tuples(triples):
        rows.append(json.dumps([subject, predicate, obj], ensure_ascii=False))
    return ", ".join(rows) if rows else "[]"


def build_pive_predict_instruction(dataset: str) -> str:
    dataset_name = "WebNLG"
    return f"Predict the missing triple given the text and graph for {dataset_name} dataset."


def parse_pive_prediction(raw_output: str) -> Dict[str, Any]:
    text = str(raw_output).strip()
    if re.fullmatch(r"(?i)\s*correct\s*\.?\s*", text):
        return {"correct": True, "triples": []}

    candidates = [text]
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())

    parsed_triples: List[Tuple[str, str, str]] = []
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except Exception:
            payload = None
        if isinstance(payload, list):
            if len(payload) == 3 and all(isinstance(item, str) for item in payload):
                parsed = core.parse_structured_triple(" | ".join(payload))
                if parsed is not None:
                    parsed_triples.append(parsed)
                    break
            elif payload and isinstance(payload[0], list):
                for row in payload:
                    if isinstance(row, list) and len(row) == 3 and all(isinstance(item, str) for item in row):
                        parsed = core.parse_structured_triple(" | ".join(row))
                        if parsed is not None:
                            parsed_triples.append(parsed)
                if parsed_triples:
                    break

    if not parsed_triples:
        regex_matches = re.findall(r'\[\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\]', text)
        for subject, predicate, obj in regex_matches:
            parsed = core.parse_structured_triple(f"{subject} | {predicate} | {obj}")
            if parsed is not None:
                parsed_triples.append(parsed)

    deduped = core.deduplicate_triplets(parsed_triples)
    return {"correct": not deduped, "triples": core.triples_to_strings(deduped)}


def load_pive_verifier():
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    peft_config = PeftConfig.from_pretrained(PIVE_VERIFIER_MODEL_ID)
    load_kwargs = {
        "torch_dtype": torch.float16,
        "device_map": "auto",
    }
    try:
        load_kwargs["load_in_8bit"] = True
        base_model = AutoModelForSeq2SeqLM.from_pretrained(peft_config.base_model_name_or_path, **load_kwargs)
    except Exception:
        load_kwargs.pop("load_in_8bit", None)
        base_model = AutoModelForSeq2SeqLM.from_pretrained(peft_config.base_model_name_or_path, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
    tokenizer.add_tokens(["<S>"])
    base_model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base_model, PIVE_VERIFIER_MODEL_ID, torch_dtype=torch.float16, device_map="auto")
    model.eval()
    return tokenizer, model


def run_pive_signal(samples: List[Dict[str, Any]], args: argparse.Namespace, output_root: Path, repair_model: str) -> List[Dict[str, Any]]:
    dataset = str(samples[0]["dataset"])
    baseline = str(samples[0]["baseline_name"])
    cache_path = signal_cache_path(output_root, "pive_missing", dataset, baseline, repair_model)
    progress_jsonl_path = signal_progress_jsonl_path(output_root, "pive_missing", dataset, baseline, repair_model)
    cached_rows = load_signal_cache(cache_path)
    if cached_rows is not None:
        return cached_rows

    lock_path = signal_lock_path(output_root, "pive_missing", dataset, baseline, repair_model)
    with exclusive_file_lock(lock_path):
        cached_rows = load_signal_cache(cache_path)
        if cached_rows is not None:
            return cached_rows

        rows_by_id: Dict[str, Dict[str, Any]] = {}
        for record in core.load_jsonl_records(progress_jsonl_path):
            sample_id = str(record.get("sample_id", ""))
            if sample_id:
                rows_by_id[sample_id] = record

        active_samples = active_iteration_samples(samples)
        for sample in samples:
            if sample.get("iteration_active", True):
                continue
            rows_by_id.setdefault(
                sample["sample_id"],
                build_skipped_signal_row(sample, raw_signal_output="", missing_triples=[]),
            )

        pending = [sample for sample in active_samples if sample["sample_id"] not in rows_by_id]
        tokenizer, model = load_pive_verifier()
        try:
            if pending:
                device = next(model.parameters()).device
                total_batches = math.ceil(len(pending) / args.pive_signal_batch_size)
                progress_iter: Iterable[Tuple[int, int]] = enumerate(range(0, len(pending), args.pive_signal_batch_size), start=1)
                if tqdm is not None:
                    progress_iter = enumerate(
                        tqdm(
                            range(0, len(pending), args.pive_signal_batch_size),
                            total=total_batches,
                            desc=f"pive-signal {dataset}/{baseline}",
                            dynamic_ncols=True,
                            leave=True,
                        ),
                        start=1,
                    )
                for _, start in progress_iter:
                    batch = pending[start : start + args.pive_signal_batch_size]
                    model_inputs = [
                        build_pive_predict_instruction(dataset) + " " + sample["source_text"] + " <S> " + triples_to_pive_graph(sample["current_triples"])
                        for sample in batch
                    ]
                    tokenized = tokenizer(
                        model_inputs,
                        return_tensors="pt",
                        truncation=True,
                        padding=True,
                    )
                    tokenized = {key: value.to(device) for key, value in tokenized.items()}
                    with torch.no_grad():
                        outputs = model.generate(
                            **tokenized,
                            max_new_tokens=args.pive_max_new_tokens,
                            temperature=0.1,
                            top_p=0.9,
                            top_k=40,
                            num_beams=4,
                            num_return_sequences=1,
                        )
                    decoded_outputs = tokenizer.batch_decode(outputs.detach().cpu(), skip_special_tokens=True)
                    for sample, raw_output in zip(batch, decoded_outputs):
                        parsed = parse_pive_prediction(raw_output)
                        record = {
                            "sample_id": sample["sample_id"],
                            "raw_signal_output": raw_output,
                            "missing_triples": parsed["triples"],
                            "flagged": not parsed["correct"],
                        }
                        rows_by_id[sample["sample_id"]] = record
                        core.append_jsonl(progress_jsonl_path, record)
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        rows = [rows_by_id[sample["sample_id"]] for sample in samples]
        core.save_json(cache_path, rows)
        return rows


def load_nli_templates(path: Path) -> Dict[str, Any]:
    if path.is_file():
        return core.load_json(path)
    return {}


def triple_to_nli_template(triple: Dict[str, str], templates: Dict[str, Any]) -> str:
    predicate = triple["predicate"]
    subject = triple["subject"]
    obj = triple["object"]
    template = templates.get(predicate)
    if template is None and predicate in NLI_TEMPLATE_REMAP:
        template = templates.get(NLI_TEMPLATE_REMAP[predicate])
    if template is None:
        template = f"The {predicate} of <subject> is <object>."
        template = re.sub(r"([a-z])([A-Z])", r"\1 \2", template)
    if isinstance(template, dict):
        template = template.get(obj, next(iter(template.values())))
    if isinstance(template, list):
        template = template[0]
    return str(template).replace("<subject>", subject).replace("<object>", obj).replace("_", " ")


def load_nli_model():
    from transformers import RobertaForSequenceClassification, RobertaTokenizer

    tokenizer = RobertaTokenizer.from_pretrained("roberta-large-mnli")
    model = RobertaForSequenceClassification.from_pretrained("roberta-large-mnli")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device


def roberta_classify_batch(tokenizer, model, device: str, premise_hypothesis_pairs: Sequence[Tuple[str, str]]) -> np.ndarray:
    inputs = tokenizer(
        [f"{premise} </s></s> {hypothesis}" for premise, hypothesis in premise_hypothesis_pairs],
        return_tensors="pt",
        truncation=True,
        padding=True,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
    return probs


def run_nli_signal(samples: List[Dict[str, Any]], args: argparse.Namespace, output_root: Path, repair_model: str) -> List[Dict[str, Any]]:
    dataset = str(samples[0]["dataset"])
    baseline = str(samples[0]["baseline_name"])
    cache_path = signal_cache_path(output_root, "nli_binary", dataset, baseline, repair_model)
    cached_rows = load_signal_cache(cache_path)
    if cached_rows is not None:
        return cached_rows

    lock_path = signal_lock_path(output_root, "nli_binary", dataset, baseline, repair_model)
    with exclusive_file_lock(lock_path):
        cached_rows = load_signal_cache(cache_path)
        if cached_rows is not None:
            return cached_rows

        templates = load_nli_templates(args.nli_template_file)
        tokenizer, model, device = load_nli_model()
        sample_state: Dict[str, Dict[str, Any]] = {}
        try:
            triple_tasks: List[Tuple[str, str, str, str]] = []
            global_tasks: List[Tuple[str, str, str]] = []
            active_samples = active_iteration_samples(samples)
            for sample in samples:
                if not sample.get("iteration_active", True):
                    sample_state[sample["sample_id"]] = {
                        "triple_scores": [],
                        "unsupported_triples": [],
                        "global_precision": 1.0,
                        "possible_missing": False,
                        "skipped_iteration": True,
                    }
                    continue
                template_sents: List[str] = []
                triple_scores: List[Dict[str, Any]] = []
                for triple_str in sample["current_triples"]:
                    triple = core.parse_pipe_triple(triple_str)
                    if triple is None:
                        continue
                    triple_dict = {"subject": triple[0], "predicate": triple[1], "object": triple[2]}
                    template = triple_to_nli_template(triple_dict, templates)
                    template_sents.append(template)
                    triple_scores.append({"triple": triple_str, "score": 0.0, "supported": True, "template": template})
                    triple_tasks.append((sample["sample_id"], triple_str, sample["source_text"], template))
                sample_state[sample["sample_id"]] = {
                    "triple_scores": triple_scores,
                    "unsupported_triples": [],
                    "global_precision": 1.0,
                    "possible_missing": False,
                }
                if template_sents:
                    global_tasks.append((sample["sample_id"], " ".join(template_sents), sample["source_text"]))

            total_batches = math.ceil(len(triple_tasks) / args.nli_signal_batch_size) if triple_tasks else 0
            progress_iter: Iterable[Tuple[int, int]] = enumerate(range(0, len(triple_tasks), args.nli_signal_batch_size), start=1)
            if tqdm is not None and triple_tasks:
                progress_iter = enumerate(
                    tqdm(
                        range(0, len(triple_tasks), args.nli_signal_batch_size),
                        total=total_batches,
                        desc=f"nli-signal {dataset}/{baseline}",
                        dynamic_ncols=True,
                        leave=True,
                    ),
                    start=1,
                )
            for _, start in progress_iter:
                batch = triple_tasks[start : start + args.nli_signal_batch_size]
                probs_batch = roberta_classify_batch(
                    tokenizer,
                    model,
                    device,
                    [(premise, hypothesis) for _, _, premise, hypothesis in batch],
                )
                for (sample_id, triple_str, _, _), probs in zip(batch, probs_batch):
                    entailment = float(probs[2])
                    supported = entailment >= args.nli_threshold
                    state = sample_state[sample_id]
                    for item in state["triple_scores"]:
                        if item["triple"] == triple_str:
                            item["score"] = entailment
                            item["supported"] = supported
                            break
                    if not supported:
                        state["unsupported_triples"].append(triple_str)

            for start in range(0, len(global_tasks), args.nli_signal_batch_size):
                batch = global_tasks[start : start + args.nli_signal_batch_size]
                probs_batch = roberta_classify_batch(
                    tokenizer,
                    model,
                    device,
                    [(premise, hypothesis) for _, premise, hypothesis in batch],
                )
                for (sample_id, _, _), probs in zip(batch, probs_batch):
                    global_precision = float(probs[2])
                    sample_state[sample_id]["global_precision"] = global_precision
                    sample_state[sample_id]["possible_missing"] = global_precision < args.nli_global_threshold
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        rows: List[Dict[str, Any]] = []
        for sample in samples:
            state = sample_state[sample["sample_id"]]
            rows.append(
                {
                    "sample_id": sample["sample_id"],
                    "unsupported_triples": state["unsupported_triples"],
                    "triple_scores": state["triple_scores"],
                    "global_precision": state["global_precision"],
                    "possible_missing": state["possible_missing"],
                    "flagged": bool(state["unsupported_triples"]) or state["possible_missing"],
                    "skipped_iteration": bool(state.get("skipped_iteration", False)),
                }
            )
        core.save_json(cache_path, rows)
        return rows


def load_factspotter_deberta_model(model_name: str):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device


def run_factspotter_signal(samples: List[Dict[str, Any]], args: argparse.Namespace, output_root: Path, repair_model: str) -> List[Dict[str, Any]]:
    dataset = str(samples[0]["dataset"])
    baseline = str(samples[0]["baseline_name"])
    cache_path = signal_cache_path(output_root, "factspotter_binary", dataset, baseline, repair_model)
    cached_rows = load_signal_cache(cache_path)
    if cached_rows is not None:
        return cached_rows

    lock_path = signal_lock_path(output_root, "factspotter_binary", dataset, baseline, repair_model)
    with exclusive_file_lock(lock_path):
        cached_rows = load_signal_cache(cache_path)
        if cached_rows is not None:
            return cached_rows

        model_name = FACTSPOTTER_HF_MODELS[args.factspotter_backend]
        tokenizer, model, device = load_factspotter_deberta_model(model_name)
        sample_state: Dict[str, Dict[str, Any]] = {}
        try:
            pair_tasks: List[Tuple[str, str, str, str]] = []
            for sample in samples:
                if not sample.get("iteration_active", True):
                    sample_state[sample["sample_id"]] = {
                        "unsupported_triples": [],
                        "triple_scores": [],
                        "skipped_iteration": True,
                    }
                    continue
                triple_scores: List[Dict[str, Any]] = []
                for triple_str in sample["current_triples"]:
                    triple = core.parse_pipe_triple(triple_str)
                    if triple is None:
                        continue
                    subject, predicate, obj = triple
                    hypothesis = f"{subject}, {predicate}, {obj}"
                    triple_scores.append({"triple": triple_str, "score": 0.0, "supported": True})
                    pair_tasks.append((sample["sample_id"], triple_str, sample["source_text"], hypothesis))
                sample_state[sample["sample_id"]] = {
                    "unsupported_triples": [],
                    "triple_scores": triple_scores,
                }

            total_batches = math.ceil(len(pair_tasks) / args.factspotter_signal_batch_size) if pair_tasks else 0
            progress_iter: Iterable[Tuple[int, int]] = enumerate(range(0, len(pair_tasks), args.factspotter_signal_batch_size), start=1)
            if tqdm is not None and pair_tasks:
                progress_iter = enumerate(
                    tqdm(
                        range(0, len(pair_tasks), args.factspotter_signal_batch_size),
                        total=total_batches,
                        desc=f"factspotter-signal {dataset}/{baseline}",
                        dynamic_ncols=True,
                        leave=True,
                    ),
                    start=1,
                )
            for _, start in progress_iter:
                batch = pair_tasks[start : start + args.factspotter_signal_batch_size]
                tokenized = tokenizer(
                    [(premise, hypothesis) for _, _, premise, hypothesis in batch],
                    truncation=True,
                    padding=True,
                    return_token_type_ids=True,
                    return_tensors="pt",
                )
                tokenized = {key: value.to(device) for key, value in tokenized.items()}
                with torch.no_grad():
                    logits = model(**tokenized).logits
                probs_batch = torch.softmax(logits, dim=1).detach().cpu().numpy()
                for (sample_id, triple_str, _, _), probs in zip(batch, probs_batch):
                    score = float(probs[0])
                    supported = score >= args.factspotter_threshold
                    state = sample_state[sample_id]
                    for item in state["triple_scores"]:
                        if item["triple"] == triple_str:
                            item["score"] = score
                            item["supported"] = supported
                            break
                    if not supported:
                        state["unsupported_triples"].append(triple_str)
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        rows: List[Dict[str, Any]] = []
        for sample in samples:
            state = sample_state[sample["sample_id"]]
            rows.append(
                {
                    "sample_id": sample["sample_id"],
                    "unsupported_triples": state["unsupported_triples"],
                    "triple_scores": state["triple_scores"],
                    "backend": args.factspotter_backend,
                    "flagged": bool(state["unsupported_triples"]),
                    "skipped_iteration": bool(state.get("skipped_iteration", False)),
                }
            )
        core.save_json(cache_path, rows)
        return rows


def make_repair_llm(args: argparse.Namespace, model_key: str):
    repair_args = core.make_repair_args(args, model_key)
    os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
    if repair_args.model_type == "qwen3" and os.environ.get("USE_HF") is None:
        os.environ["USE_HF"] = "1"
    core.configure_multiprocessing()
    core.set_seed(args.repair_seed)
    llm = core.load_vllm(repair_args, repair_args.model_id_or_path)
    tokenizer = llm.get_tokenizer()
    sampling_params = core.build_sampling_params(repair_args, tokenizer)
    return repair_args, llm, tokenizer, sampling_params


def generate_raw_texts(
    *,
    llm,
    tokenizer,
    repair_args,
    sampling_params,
    message_batches: Sequence[List[Dict[str, str]]],
) -> List[str]:
    prompts = [core.render_prompt(repair_args, tokenizer, messages) for messages in message_batches]
    outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
    return [output.outputs[0].text for output in outputs]


def build_pive_repair_prompt(source_text: str, missing_triples: Sequence[str]) -> str:
    return (
        PIVE_REPAIR_EXAMPLE
        + f"Text: {source_text}\n"
        + f"Triples: {triples_to_pive_inline(missing_triples)}\n"
        + "Semantic graph:"
    )


def build_self_refine_feedback_prompt_v2(source_text: str, current_triples: Sequence[str]) -> str:
    return f"""Review the extracted triples against the source text.

# Source text
{source_text}

# Current triples
{core.markdown_from_strings(current_triples)}

# Task
- Identify only high-confidence faithfulness issues.
- Report an unsupported triple only when the source text does not support it.
- Report an incorrect triple only when the source text supports a conflicting fact.
- Report a missing triple only when the source text explicitly states the fact and the omission materially reduces factual faithfulness.

# Decision rule
- Do not report issues for paraphrase, wording preference, harmless splitting or merging of equivalent facts, or facts supported by clear implication.
- If you cannot point to a concrete faithfulness error, output exactly: NO_ISSUES

# Output requirements
- If issues are present, output three sections named exactly:
Unsupported:
Incorrect:
Missing:
- Under each section, write one candidate triple per bullet in `subject | predicate | object` form when possible.
- Do not output a markdown table.
"""


SELF_REFINE_FEEDBACK_PROMPT_BUILDERS = {
    "self_refine_v2": build_self_refine_feedback_prompt_v2,
}


def build_self_refine_repair_prompt(source_text: str, current_triples: Sequence[str], feedback_text: str) -> str:
    return f"""Repair the extracted triples using the source text and self feedback.

# Source text
{source_text}

# Current triples
{core.markdown_from_strings(current_triples)}

# Self feedback
{feedback_text}

# Output requirements
- Return only a markdown table with exactly three columns: subject, predicate, object.
- Keep one row per final triple.
- Use the source text as the final authority.
- Keep unaffected supported triples unchanged.
- Do not explain your reasoning.
- If no triples remain, return only the header row and separator row.
"""


def build_nli_repair_prompt(
    source_text: str,
    current_triples: Sequence[str],
    unsupported_triples: Sequence[str],
    possible_missing: bool,
    global_precision: float,
) -> str:
    missing_note = (
        f"- Global NLI graph-to-text support score is {global_precision:.3f}, so the text may contain additional supported facts beyond the current triples.\n"
        if possible_missing
        else "- Global NLI check does not strongly suggest additional missing facts.\n"
    )
    return f"""Repair the extracted triples using the source text and NLI-based support checks.

# Source text
{source_text}

# Current triples
{core.markdown_from_strings(current_triples)}

# Triples marked as not supported by NLI
{core.markdown_from_strings(unsupported_triples) if unsupported_triples else '- None'}

# Coarse global NLI note
{missing_note}

# Task
- Revise or remove triples that are not supported by the text.
- If the text clearly states a missing fact, you may add it.
- Keep unaffected supported triples unchanged.
- Use the source text as the final authority.

# Output requirements
- Return only a markdown table with exactly three columns: subject, predicate, object.
- Keep one row per final triple.
- Do not explain your reasoning.
- If no triples remain, return only the header row and separator row.
"""


def build_factspotter_repair_prompt(
    source_text: str,
    current_triples: Sequence[str],
    unsupported_triples: Sequence[str],
) -> str:
    return f"""Repair the extracted triples using the source text and per-triple FactSpotter support checks.

# Source text
{source_text}

# Current triples
{core.markdown_from_strings(current_triples)}

# Triples marked as not supported by FactSpotter
{core.markdown_from_strings(unsupported_triples) if unsupported_triples else '- None'}

# Task
- Revise or remove triples that are not supported by the text.
- Do not assume this signal has identified all missing facts.
- Add a new triple only when the text clearly and explicitly states a fact that is needed to produce a faithful final table.
- Keep unaffected supported triples unchanged.

# Output requirements
- Return only a markdown table with exactly three columns: subject, predicate, object.
- Keep one row per final triple.
- Do not explain your reasoning.
- If no triples remain, return only the header row and separator row.
"""


def finalize_entries(
    *,
    samples: List[Dict[str, Any]],
    repaired_by_id: Dict[str, List[str]],
    records_by_id: Dict[str, Dict[str, Any]],
    output_path: Path,
    dataset: str,
    baseline: str,
    repair_model: str,
    mode: str,
    output_root: Path,
) -> Path:
    shot_k = int(samples[0].get("shot_k", 0))
    changed = 0
    flagged = 0
    entries: List[Dict[str, Any]] = []
    for sample in samples:
        output_triples = repaired_by_id.get(sample["sample_id"], list(sample["current_triples"]))
        record = records_by_id.get(sample["sample_id"], {})
        if record:
            flagged += 1
        if output_triples != sample["current_triples"]:
            changed += 1
        entries.append(
            {
                "sample_id": sample["sample_id"],
                "triplets": core.strings_to_dicts(output_triples),
                "raw_output": str(record.get("raw_model_output", "")),
                "source_text": sample["source_text"],
                "baseline_name": baseline,
                "repair_model": repair_model,
                "prompt_variant": mode,
                "flagged": bool(record),
            }
        )

    payload = core.build_entries_payload(
        dataset=dataset,
        model_label=f"{repair_model}_{baseline}_{mode}",
        prompt_variant=mode,
        baseline=baseline,
        shot_k=shot_k,
        entries=entries,
        extra_meta={
            "repair_model": repair_model,
            "verifier_model": mode,
            "verifier_prompt": mode,
        },
    )
    core.save_json(output_path, payload)
    summary_path = output_root / "repair_records" / dataset / baseline / repair_model / mode / "summary.json"
    core.save_json(
        summary_path,
        {
            "dataset": dataset,
            "baseline_name": baseline,
            "repair_model": repair_model,
            "prompt_variant": mode,
            "verifier_model": mode,
            "num_samples": len(samples),
            "flagged_samples": flagged,
            "changed_samples": changed,
        },
    )
    return output_path


def run_pive_mode(
    *,
    samples: List[Dict[str, Any]],
    signal_rows: List[Dict[str, Any]],
    repair_model: str,
    args: argparse.Namespace,
    output_root: Path,
) -> Path:
    dataset = str(samples[0]["dataset"])
    baseline = str(samples[0]["baseline_name"])
    shot_k = int(samples[0].get("shot_k", 0))
    mode = "pive_missing"
    output_dir = core.ensure_dir(output_root / "markdown" / dataset)
    output_path = output_dir / baseline_output_filename(dataset, baseline, repair_model, mode, shot_k)
    if output_path.exists():
        return output_path

    repair_records_dir = core.ensure_dir(output_root / "repair_records" / dataset / baseline / repair_model / mode)
    progress_jsonl_path = repair_records_dir / "records.jsonl"
    signal_by_id = {row["sample_id"]: row for row in signal_rows}
    repaired_by_id: Dict[str, List[str]] = {}
    records_by_id: Dict[str, Dict[str, Any]] = {}
    for record in core.load_jsonl_records(progress_jsonl_path):
        sample_id = str(record.get("sample_id", ""))
        parsed_triples = record.get("parsed_triples")
        if sample_id and isinstance(parsed_triples, list):
            repaired_by_id[sample_id] = [str(item) for item in parsed_triples]
            records_by_id[sample_id] = record

    pending = [
        sample
        for sample in samples
        if signal_by_id[sample["sample_id"]]["flagged"] and sample["sample_id"] not in repaired_by_id
    ]

    if pending:
        repair_args, llm, tokenizer, sampling_params = make_repair_llm(args, repair_model)
        try:
            total_batches = math.ceil(len(pending) / args.repair_batch_size)
            progress_iter: Iterable[Tuple[int, int]] = enumerate(range(0, len(pending), args.repair_batch_size), start=1)
            if tqdm is not None:
                progress_iter = enumerate(
                    tqdm(
                        range(0, len(pending), args.repair_batch_size),
                        total=total_batches,
                        desc=f"repair {dataset}/{baseline}/{repair_model}/{mode}",
                        dynamic_ncols=True,
                        leave=True,
                    ),
                    start=1,
                )
            for _, start in progress_iter:
                batch = pending[start : start + args.repair_batch_size]
                messages = [
                    [{"role": "user", "content": build_pive_repair_prompt(sample["source_text"], signal_by_id[sample["sample_id"]]["missing_triples"])}]
                    for sample in batch
                ]
                raw_outputs = generate_raw_texts(
                    llm=llm,
                    tokenizer=tokenizer,
                    repair_args=repair_args,
                    sampling_params=sampling_params,
                    message_batches=messages,
                )
                for sample, raw_output in zip(batch, raw_outputs):
                    parsed_triples = core.parse_repair_output(raw_output)
                    fell_back = False
                    if not parsed_triples:
                        parsed_triples = list(sample["current_triples"])
                        fell_back = True
                    record = {
                        "sample_id": sample["sample_id"],
                        "raw_signal_output": signal_by_id[sample["sample_id"]]["raw_signal_output"],
                        "missing_triples": signal_by_id[sample["sample_id"]]["missing_triples"],
                        "raw_model_output": raw_output,
                        "parsed_triples": parsed_triples,
                        "fell_back_to_previous": fell_back,
                    }
                    repaired_by_id[sample["sample_id"]] = parsed_triples
                    records_by_id[sample["sample_id"]] = record
                    core.append_jsonl(progress_jsonl_path, record)
        finally:
            del llm
            gc.collect()
            core.verifier.clear_cuda_memory()

    for sample in samples:
        if sample["sample_id"] not in repaired_by_id:
            repaired_by_id[sample["sample_id"]] = list(sample["current_triples"])

    core.save_json(repair_records_dir / "records.json", list(records_by_id.values()))
    return finalize_entries(
        samples=samples,
        repaired_by_id=repaired_by_id,
        records_by_id=records_by_id,
        output_path=output_path,
        dataset=dataset,
        baseline=baseline,
        repair_model=repair_model,
        mode=mode,
        output_root=output_root,
    )


def run_self_refine_mode(
    *,
    samples: List[Dict[str, Any]],
    repair_model: str,
    args: argparse.Namespace,
    output_root: Path,
    mode: str = "self_refine",
) -> Path:
    dataset = str(samples[0]["dataset"])
    baseline = str(samples[0]["baseline_name"])
    shot_k = int(samples[0].get("shot_k", 0))
    output_dir = core.ensure_dir(output_root / "markdown" / dataset)
    output_path = output_dir / baseline_output_filename(dataset, baseline, repair_model, mode, shot_k)
    if output_path.exists():
        return output_path

    repair_records_dir = core.ensure_dir(output_root / "repair_records" / dataset / baseline / repair_model / mode)
    feedback_jsonl_path = repair_records_dir / "feedback_records.jsonl"
    progress_jsonl_path = repair_records_dir / "records.jsonl"
    feedback_by_id: Dict[str, Dict[str, Any]] = {}
    for record in core.load_jsonl_records(feedback_jsonl_path):
        sample_id = str(record.get("sample_id", ""))
        raw_feedback_output = record.get("raw_feedback_output")
        if sample_id and isinstance(raw_feedback_output, str):
            feedback_by_id[sample_id] = record

    repaired_by_id: Dict[str, List[str]] = {}
    records_by_id: Dict[str, Dict[str, Any]] = {}
    for record in core.load_jsonl_records(progress_jsonl_path):
        sample_id = str(record.get("sample_id", ""))
        parsed_triples = record.get("parsed_triples")
        if sample_id and isinstance(parsed_triples, list):
            repaired_by_id[sample_id] = [str(item) for item in parsed_triples]
            records_by_id[sample_id] = record

    active_samples = active_iteration_samples(samples)
    for sample in samples:
        if not sample.get("iteration_active", True):
            repaired_by_id.setdefault(sample["sample_id"], list(sample["current_triples"]))
            continue
        sample_id = sample["sample_id"]
        feedback_record = feedback_by_id.get(sample_id)
        if sample_id in repaired_by_id or feedback_record is None:
            continue
        feedback_text = str(feedback_record["raw_feedback_output"]).strip()
        if feedback_text != "NO_ISSUES":
            continue
        record = {
            "sample_id": sample_id,
            "raw_feedback_output": feedback_text,
            "raw_model_output": "",
            "parsed_triples": list(sample["current_triples"]),
            "fell_back_to_previous": False,
            "used_repair_step": False,
        }
        repaired_by_id[sample_id] = list(sample["current_triples"])
        records_by_id[sample_id] = record
        core.append_jsonl(progress_jsonl_path, record)

    feedback_pending = [sample for sample in active_samples if sample["sample_id"] not in feedback_by_id and sample["sample_id"] not in repaired_by_id]
    repair_pending = [sample for sample in active_samples if sample["sample_id"] in feedback_by_id and sample["sample_id"] not in repaired_by_id]
    if feedback_pending or repair_pending:
        feedback_prompt_builder = SELF_REFINE_FEEDBACK_PROMPT_BUILDERS[mode]
        repair_args, llm, tokenizer, sampling_params = make_repair_llm(args, repair_model)
        try:
            if feedback_pending:
                total_batches = math.ceil(len(feedback_pending) / args.self_refine_feedback_batch_size)
                progress_iter: Iterable[Tuple[int, int]] = enumerate(range(0, len(feedback_pending), args.self_refine_feedback_batch_size), start=1)
                if tqdm is not None:
                    progress_iter = enumerate(
                        tqdm(
                            range(0, len(feedback_pending), args.self_refine_feedback_batch_size),
                            total=total_batches,
                            desc=f"feedback {dataset}/{baseline}/{repair_model}/{mode}",
                            dynamic_ncols=True,
                            leave=True,
                        ),
                        start=1,
                    )
                for _, start in progress_iter:
                    batch = feedback_pending[start : start + args.self_refine_feedback_batch_size]
                    feedback_messages = [
                        [
                            {"role": "system", "content": SELF_REFINE_FEEDBACK_SYSTEM},
                            {"role": "user", "content": feedback_prompt_builder(sample["source_text"], sample["current_triples"])},
                        ]
                        for sample in batch
                    ]
                    feedback_outputs = generate_raw_texts(
                        llm=llm,
                        tokenizer=tokenizer,
                        repair_args=repair_args,
                        sampling_params=sampling_params,
                        message_batches=feedback_messages,
                    )
                    for sample, feedback_text in zip(batch, feedback_outputs):
                        feedback_record = {
                            "sample_id": sample["sample_id"],
                            "raw_feedback_output": feedback_text,
                        }
                        feedback_by_id[sample["sample_id"]] = feedback_record
                        core.append_jsonl(feedback_jsonl_path, feedback_record)
                        if feedback_text.strip() != "NO_ISSUES":
                            continue
                        record = {
                            "sample_id": sample["sample_id"],
                            "raw_feedback_output": feedback_text,
                            "raw_model_output": "",
                            "parsed_triples": list(sample["current_triples"]),
                            "fell_back_to_previous": False,
                            "used_repair_step": False,
                        }
                        repaired_by_id[sample["sample_id"]] = list(sample["current_triples"])
                        records_by_id[sample["sample_id"]] = record
                        core.append_jsonl(progress_jsonl_path, record)

            repair_batch_items = []
            for sample in active_samples:
                sample_id = sample["sample_id"]
                if sample_id in repaired_by_id:
                    continue
                feedback_record = feedback_by_id.get(sample_id)
                if feedback_record is None:
                    continue
                feedback_text = str(feedback_record["raw_feedback_output"])
                if feedback_text.strip() == "NO_ISSUES":
                    continue
                repair_messages = [
                    {"role": "system", "content": SELF_REFINE_REPAIR_SYSTEM},
                    {"role": "user", "content": build_self_refine_repair_prompt(sample["source_text"], sample["current_triples"], feedback_text)},
                ]
                repair_batch_items.append((sample, feedback_text, repair_messages))

            total_batches = math.ceil(len(repair_batch_items) / args.repair_batch_size) if repair_batch_items else 0
            progress_iter = enumerate(range(0, len(repair_batch_items), args.repair_batch_size), start=1)
            if tqdm is not None and repair_batch_items:
                progress_iter = enumerate(
                    tqdm(
                        range(0, len(repair_batch_items), args.repair_batch_size),
                        total=total_batches,
                        desc=f"repair {dataset}/{baseline}/{repair_model}/{mode}",
                        dynamic_ncols=True,
                        leave=True,
                    ),
                    start=1,
                )
            for _, start in progress_iter:
                batch = repair_batch_items[start : start + args.repair_batch_size]
                raw_outputs = generate_raw_texts(
                    llm=llm,
                    tokenizer=tokenizer,
                    repair_args=repair_args,
                    sampling_params=sampling_params,
                    message_batches=[item[2] for item in batch],
                )
                for (sample, feedback_text, _), raw_output in zip(batch, raw_outputs):
                    parsed_triples = core.parse_repair_output(raw_output)
                    fell_back = False
                    if not parsed_triples:
                        parsed_triples = list(sample["current_triples"])
                        fell_back = True
                    record = {
                        "sample_id": sample["sample_id"],
                        "raw_feedback_output": feedback_text,
                        "raw_model_output": raw_output,
                        "parsed_triples": parsed_triples,
                        "fell_back_to_previous": fell_back,
                        "used_repair_step": True,
                    }
                    repaired_by_id[sample["sample_id"]] = parsed_triples
                    records_by_id[sample["sample_id"]] = record
                    core.append_jsonl(progress_jsonl_path, record)
        finally:
            del llm
            gc.collect()
            core.verifier.clear_cuda_memory()

    core.save_json(repair_records_dir / "feedback_records.json", list(feedback_by_id.values()))
    core.save_json(repair_records_dir / "records.json", list(records_by_id.values()))
    return finalize_entries(
        samples=samples,
        repaired_by_id=repaired_by_id,
        records_by_id=records_by_id,
        output_path=output_path,
        dataset=dataset,
        baseline=baseline,
        repair_model=repair_model,
        mode=mode,
        output_root=output_root,
    )


def run_signal_repair_mode(
    *,
    samples: List[Dict[str, Any]],
    signal_rows: List[Dict[str, Any]],
    repair_model: str,
    args: argparse.Namespace,
    output_root: Path,
    mode: str,
) -> Path:
    dataset = str(samples[0]["dataset"])
    baseline = str(samples[0]["baseline_name"])
    shot_k = int(samples[0].get("shot_k", 0))
    output_dir = core.ensure_dir(output_root / "markdown" / dataset)
    output_path = output_dir / baseline_output_filename(dataset, baseline, repair_model, mode, shot_k)
    if output_path.exists():
        return output_path

    repair_records_dir = core.ensure_dir(output_root / "repair_records" / dataset / baseline / repair_model / mode)
    progress_jsonl_path = repair_records_dir / "records.jsonl"
    signal_by_id = {row["sample_id"]: row for row in signal_rows}
    repaired_by_id: Dict[str, List[str]] = {}
    records_by_id: Dict[str, Dict[str, Any]] = {}
    for record in core.load_jsonl_records(progress_jsonl_path):
        sample_id = str(record.get("sample_id", ""))
        parsed_triples = record.get("parsed_triples")
        if sample_id and isinstance(parsed_triples, list):
            repaired_by_id[sample_id] = [str(item) for item in parsed_triples]
            records_by_id[sample_id] = record

    pending = [
        sample
        for sample in samples
        if signal_by_id[sample["sample_id"]]["flagged"] and sample["sample_id"] not in repaired_by_id
    ]
    if pending:
        repair_args, llm, tokenizer, sampling_params = make_repair_llm(args, repair_model)
        try:
            total_batches = math.ceil(len(pending) / args.repair_batch_size)
            progress_iter: Iterable[Tuple[int, int]] = enumerate(range(0, len(pending), args.repair_batch_size), start=1)
            if tqdm is not None:
                progress_iter = enumerate(
                    tqdm(
                        range(0, len(pending), args.repair_batch_size),
                        total=total_batches,
                        desc=f"repair {dataset}/{baseline}/{repair_model}/{mode}",
                        dynamic_ncols=True,
                        leave=True,
                    ),
                    start=1,
                )
            for _, start in progress_iter:
                batch = pending[start : start + args.repair_batch_size]
                messages = []
                for sample in batch:
                    signal = signal_by_id[sample["sample_id"]]
                    if mode == "nli_binary":
                        prompt = build_nli_repair_prompt(
                            sample["source_text"],
                            sample["current_triples"],
                            signal["unsupported_triples"],
                            signal["possible_missing"],
                            float(signal["global_precision"]),
                        )
                    elif mode == "factspotter_binary":
                        prompt = build_factspotter_repair_prompt(
                            sample["source_text"],
                            sample["current_triples"],
                            signal["unsupported_triples"],
                        )
                    else:  # pragma: no cover
                        raise ValueError(f"Unsupported mode: {mode}")
                    messages.append(
                        [
                            {"role": "system", "content": core.REPAIR_SYSTEM},
                            {"role": "user", "content": prompt},
                        ]
                    )
                raw_outputs = generate_raw_texts(
                    llm=llm,
                    tokenizer=tokenizer,
                    repair_args=repair_args,
                    sampling_params=sampling_params,
                    message_batches=messages,
                )
                for sample, raw_output in zip(batch, raw_outputs):
                    parsed_triples = core.parse_repair_output(raw_output)
                    fell_back = False
                    if not parsed_triples:
                        parsed_triples = list(sample["current_triples"])
                        fell_back = True
                    record = {
                        "sample_id": sample["sample_id"],
                        "signal_row": signal_by_id[sample["sample_id"]],
                        "raw_model_output": raw_output,
                        "parsed_triples": parsed_triples,
                        "fell_back_to_previous": fell_back,
                    }
                    repaired_by_id[sample["sample_id"]] = parsed_triples
                    records_by_id[sample["sample_id"]] = record
                    core.append_jsonl(progress_jsonl_path, record)
        finally:
            del llm
            gc.collect()
            core.verifier.clear_cuda_memory()

    for sample in samples:
        if sample["sample_id"] not in repaired_by_id:
            repaired_by_id[sample["sample_id"]] = list(sample["current_triples"])

    core.save_json(repair_records_dir / "records.json", list(records_by_id.values()))
    return finalize_entries(
        samples=samples,
        repaired_by_id=repaired_by_id,
        records_by_id=records_by_id,
        output_path=output_path,
        dataset=dataset,
        baseline=baseline,
        repair_model=repair_model,
        mode=mode,
        output_root=output_root,
    )


def main() -> None:
    args = parse_args()
    output_root = core.ensure_dir(args.output_root)
    if args.input_json is not None and len(args.datasets) * len(args.baselines) != 1:
        raise ValueError("--input-json requires exactly one dataset and one baseline per invocation.")

    prepared: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for dataset in args.datasets:
        for baseline in args.baselines:
            samples = core.load_baseline_samples(
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
            if args.input_json is None:
                baseline_path = core.save_baseline_payload(samples, output_root)
                print(f"[BASELINE] wrote {baseline_path}")

    for mode in args.modes:
        for dataset, baseline in prepared:
            samples = prepared[(dataset, baseline)]
            if mode == "pive_missing":
                for repair_model in args.repair_models:
                    signal_rows = run_pive_signal(samples, args, output_root, repair_model)
                    flagged = sum(1 for row in signal_rows if row["flagged"])
                    print(
                        f"[SIGNAL] mode={mode} dataset={dataset} baseline={baseline} "
                        f"repair={repair_model} flagged={flagged}/{len(signal_rows)}"
                    )
                    output_path = run_pive_mode(
                        samples=samples,
                        signal_rows=signal_rows,
                        repair_model=repair_model,
                        args=args,
                        output_root=output_root,
                    )
                    print(f"[REPAIR] mode={mode} dataset={dataset} baseline={baseline} repair={repair_model} -> {output_path}")
            elif mode == "self_refine_v2":
                for repair_model in args.repair_models:
                    output_path = run_self_refine_mode(
                        samples=samples,
                        repair_model=repair_model,
                        args=args,
                        output_root=output_root,
                        mode=mode,
                    )
                    print(f"[REPAIR] mode={mode} dataset={dataset} baseline={baseline} repair={repair_model} -> {output_path}")
            elif mode == "nli_binary":
                for repair_model in args.repair_models:
                    signal_rows = run_nli_signal(samples, args, output_root, repair_model)
                    flagged = sum(1 for row in signal_rows if row["flagged"])
                    print(
                        f"[SIGNAL] mode={mode} dataset={dataset} baseline={baseline} "
                        f"repair={repair_model} flagged={flagged}/{len(signal_rows)}"
                    )
                    output_path = run_signal_repair_mode(
                        samples=samples,
                        signal_rows=signal_rows,
                        repair_model=repair_model,
                        args=args,
                        output_root=output_root,
                        mode=mode,
                    )
                    print(f"[REPAIR] mode={mode} dataset={dataset} baseline={baseline} repair={repair_model} -> {output_path}")
            elif mode == "factspotter_binary":
                for repair_model in args.repair_models:
                    signal_rows = run_factspotter_signal(samples, args, output_root, repair_model)
                    flagged = sum(1 for row in signal_rows if row["flagged"])
                    print(
                        f"[SIGNAL] mode={mode} dataset={dataset} baseline={baseline} "
                        f"repair={repair_model} flagged={flagged}/{len(signal_rows)}"
                    )
                    output_path = run_signal_repair_mode(
                        samples=samples,
                        signal_rows=signal_rows,
                        repair_model=repair_model,
                        args=args,
                        output_root=output_root,
                        mode=mode,
                    )
                    print(f"[REPAIR] mode={mode} dataset={dataset} baseline={baseline} repair={repair_model} -> {output_path}")
            else:  # pragma: no cover
                raise ValueError(f"Unsupported mode: {mode}")


if __name__ == "__main__":
    main()
