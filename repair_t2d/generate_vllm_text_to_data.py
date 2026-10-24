from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

REPAIR_ROOT = Path(__file__).resolve().parent
REPO_ROOT = REPAIR_ROOT.parent
if str(REPAIR_ROOT) not in sys.path:
    sys.path.insert(0, str(REPAIR_ROOT))

from generate_vllm_simple import (  # noqa: E402
    append_prediction,
    build_sampling_params,
    configure_multiprocessing,
    load_completed_predictions,
    load_vllm,
    render_prompt,
)


DEFAULT_SELECTOR_CACHE_DIR = REPAIR_ROOT / ".cache_text_to_data_selector"
DATASET_MAX_TRIPLETS = {
    "webnlg": 7,
    "genwiki": 14,
}


def clean_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_camel(text: str) -> str:
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)


def normalize_surface(text: str) -> str:
    value = text.replace("_", " ")
    value = re.sub(r"[\"'`“”‘’]", "", value)
    return clean_whitespace(value)


def normalize_predicate(text: str) -> str:
    value = split_camel(normalize_surface(text).replace("-", " "))
    return clean_whitespace(value).lower()


def get_output_filename(args: argparse.Namespace, suffix: str) -> str:
    if args.output_filename:
        stem = args.output_filename
    else:
        input_stem = Path(args.input_file).stem
        stem = f"{input_stem}_{args.output_style}_{args.shot_k}shot"
    return str(Path(args.output_dir) / f"{stem}.{suffix}")


def get_entry_key(sample: Dict, idx: int) -> str:
    return str(sample.get("sample_id") or f"idx_{idx}")


def load_unified_samples(path: Path) -> List[Dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Unified test file must be a list: {path}")
    return payload


def get_all_lexicalisations(entry: Dict, lang: str = "en") -> List[str]:
    lexicalisations = entry.get("lexicalisations", {})
    lang_lexes = lexicalisations.get(lang) or lexicalisations.get("en") or []
    texts: List[str] = []
    seen = set()
    for item in lang_lexes:
        if isinstance(item, dict):
            text = clean_whitespace(str(item.get("lex") or item.get("comment") or ""))
        else:
            text = clean_whitespace(str(item))
        if text and text not in seen:
            texts.append(text)
            seen.add(text)
    return texts


def entry_triplets(entry: Dict) -> List[Tuple[str, str, str]]:
    triplets: List[Tuple[str, str, str]] = []
    for triple in entry.get("modifiedtripleset", []):
        triplets.append(
            (
                normalize_surface(str(triple.get("subject", ""))),
                normalize_predicate(str(triple.get("property", ""))),
                normalize_surface(str(triple.get("object", ""))),
            )
        )
    return deduplicate_triplets(triplets)


def deduplicate_triplets(triples: Iterable[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
    deduped: List[Tuple[str, str, str]] = []
    seen = set()
    for subject, predicate, obj in triples:
        triple = (
            normalize_surface(subject),
            normalize_predicate(predicate),
            normalize_surface(obj),
        )
        if not all(triple):
            continue
        if triple in seen:
            continue
        seen.add(triple)
        deduped.append(triple)
    return deduped


@dataclass
class RetrievalExample:
    sample_id: str
    source_text: str
    triplets: List[Tuple[str, str, str]]


class PapalucaRetriever:
    def __init__(
        self,
        train_path: Path,
        fewshot_lang: str,
        selector_model_name: str,
        selector_device: str,
        selector_cache_dir: Path,
    ) -> None:
        self.examples = self._load_examples(train_path, fewshot_lang)
        selector_cache_dir.mkdir(parents=True, exist_ok=True)

        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            selector_model_name,
            device=selector_device,
            cache_folder=str(selector_cache_dir),
        )
        self.embeddings = np.asarray(
            self.model.encode(
                [example.source_text for example in self.examples],
                normalize_embeddings=True,
                show_progress_bar=True,
            ),
            dtype=np.float32,
        )

    def _load_examples(self, train_path: Path, fewshot_lang: str) -> List[RetrievalExample]:
        payload = json.loads(train_path.read_text(encoding="utf-8"))
        if "entries" not in payload:
            raise ValueError(f"Few-shot source missing 'entries': {train_path}")

        examples: List[RetrievalExample] = []
        for graph_idx, entry in enumerate(payload["entries"]):
            triplets = entry_triplets(entry)
            if not triplets:
                continue
            lexicalisations = get_all_lexicalisations(entry, fewshot_lang)
            base_sample_id = str(entry.get("id_short") or entry.get("xml_id") or f"graph_{graph_idx}")
            for lex_idx, source_text in enumerate(lexicalisations):
                examples.append(
                    RetrievalExample(
                        sample_id=f"{base_sample_id}#lex{lex_idx}",
                        source_text=source_text,
                        triplets=triplets,
                    )
                )
        if not examples:
            raise ValueError(f"No usable few-shot examples found in {train_path}")
        return examples

    def retrieve(self, query_text: str, top_k: int) -> List[RetrievalExample]:
        if top_k <= 0:
            return []
        if top_k >= len(self.examples):
            top_indices = list(range(len(self.examples)))
        else:
            query_embedding = np.asarray(
                self.model.encode(
                    [query_text],
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )[0],
                dtype=np.float32,
            )
            scores = self.embeddings @ query_embedding
            candidate_indices = np.argpartition(scores, -top_k)[-top_k:]
            top_indices = sorted(candidate_indices.tolist(), key=lambda idx: float(scores[idx]), reverse=True)
        return [self.examples[idx] for idx in top_indices[:top_k]]


class GPUMemoryMonitor:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self._nvml = None
        self._handles = []
        self._labels: List[str] = []
        self.error: Optional[str] = None

        if not enabled:
            return

        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            device_ids = self._resolve_visible_device_ids(pynvml)
            for device_id in device_ids:
                self._handles.append(pynvml.nvmlDeviceGetHandleByIndex(device_id))
                self._labels.append(f"gpu{device_id}")
        except Exception as exc:  # pragma: no cover - best effort telemetry
            self.error = str(exc)
            self.enabled = False
            self._nvml = None
            self._handles = []
            self._labels = []

    @staticmethod
    def _resolve_visible_device_ids(pynvml) -> List[int]:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if not visible:
            return list(range(pynvml.nvmlDeviceGetCount()))

        device_ids: List[int] = []
        for item in visible.split(","):
            item = item.strip()
            if not item:
                continue
            if item.isdigit():
                device_ids.append(int(item))
        if device_ids:
            return device_ids
        return list(range(pynvml.nvmlDeviceGetCount()))

    def format_postfix(self) -> str:
        if not self.enabled or not self._nvml or not self._handles:
            return ""

        parts = []
        for label, handle in zip(self._labels, self._handles):
            info = self._nvml.nvmlDeviceGetMemoryInfo(handle)
            used_gib = info.used / (1024 ** 3)
            total_gib = info.total / (1024 ** 3)
            parts.append(f"{label} {used_gib:.1f}/{total_gib:.0f}G")
        return " | ".join(parts)


def set_progress_postfix(loop, status: str, gpu_mem_monitor: Optional[GPUMemoryMonitor]) -> None:
    parts = [status]
    if gpu_mem_monitor is not None:
        mem_status = gpu_mem_monitor.format_postfix()
        if mem_status:
            parts.append(mem_status)
    loop.set_postfix_str(" | ".join(part for part in parts if part))


def escape_markdown_cell(text: str) -> str:
    return clean_whitespace(text).replace("|", "\\|")


def format_triplets_json(triples: Sequence[Tuple[str, str, str]]) -> str:
    payload = {
        "triplets": [
            {
                "subject": subject,
                "predicate": predicate,
                "object": obj,
            }
            for subject, predicate, obj in triples
        ]
    }
    return json.dumps(payload, ensure_ascii=False)


def format_triplets_markdown(triples: Sequence[Tuple[str, str, str]]) -> str:
    lines = [
        "| subject | predicate | object |",
        "|---|---|---|",
    ]
    for subject, predicate, obj in triples:
        lines.append(
            f"| {escape_markdown_cell(subject)} | {escape_markdown_cell(predicate)} | {escape_markdown_cell(obj)} |"
        )
    return "\n".join(lines)


def format_triplets_payload(triples: Sequence[Tuple[str, str, str]], output_style: str) -> str:
    if output_style == "json":
        return format_triplets_json(triples)
    if output_style == "markdown":
        return format_triplets_markdown(triples)
    raise ValueError(f"Unsupported output_style: {output_style}")


def infer_max_knowledge_triplets(args: argparse.Namespace) -> int:
    if args.max_knowledge_triplets is not None:
        return args.max_knowledge_triplets
    dataset_name = args.dataset_name.lower()
    if dataset_name in DATASET_MAX_TRIPLETS:
        return DATASET_MAX_TRIPLETS[dataset_name]
    input_name = Path(args.input_file).name.lower()
    for dataset_name, max_triplets in DATASET_MAX_TRIPLETS.items():
        if dataset_name in input_name:
            return max_triplets
    raise ValueError("Unable to infer max_knowledge_triplets; pass --max_knowledge_triplets explicitly.")


def build_prompt(
    source_text: str,
    fewshots: Sequence[RetrievalExample],
    output_style: str,
    max_knowledge_triplets: int,
) -> str:
    if output_style == "json":
        body = (
            f"Some text is provided below. Given the text, extract up to {max_knowledge_triplets} "
            "knowledge triplets with fields subject, predicate, and object. Avoid stopwords in predicates. "
            'Return only JSON with a single top-level key "triplets". Each item in "triplets" must be an object '
            'with keys "subject", "predicate", and "object". If no triplets are present, return {"triplets":[]}.'
        )
    elif output_style == "markdown":
        body = (
            f"Some text is provided below. Given the text, extract up to {max_knowledge_triplets} "
            "knowledge triplets with fields subject, predicate, and object. Avoid stopwords in predicates. "
            "Return only a markdown table with exactly three columns: subject, predicate, object. "
            "If no triplets are present, return only the table header and separator row."
        )
    else:
        raise ValueError(f"Unsupported output_style: {output_style}")

    parts = [body, "---------------------", "Examples:"]
    for example in fewshots:
        parts.append(
            f"Text: {example.source_text}\n"
            f"Triplets:\n{format_triplets_payload(example.triplets, output_style)}"
        )
    parts.append("---------------------")
    parts.append(f"Text: {source_text}\nTriplets:")
    return "\n".join(parts)


def strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[A-Za-z0-9_-]*\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def parse_triplets_json(raw_output: str) -> List[Tuple[str, str, str]]:
    text = strip_code_fences(raw_output)
    candidates = [text]
    if "{" in text and "}" in text:
        candidates.append(text[text.find("{") : text.rfind("}") + 1])
    if "[" in text and "]" in text:
        candidates.append(text[text.find("[") : text.rfind("]") + 1])

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue

        if isinstance(payload, dict):
            triplets_payload = payload.get("triplets", [])
        elif isinstance(payload, list):
            triplets_payload = payload
        else:
            continue

        parsed: List[Tuple[str, str, str]] = []
        for item in triplets_payload:
            if isinstance(item, dict):
                subject = str(item.get("subject", ""))
                predicate = str(item.get("predicate", ""))
                obj = str(item.get("object", ""))
            elif isinstance(item, (list, tuple)) and len(item) == 3:
                subject, predicate, obj = map(str, item)
            else:
                continue
            parsed.append((subject, predicate, obj))
        return deduplicate_triplets(parsed)

    return []


def parse_triplets_markdown(raw_output: str) -> List[Tuple[str, str, str]]:
    text = strip_code_fences(raw_output)
    parsed: List[Tuple[str, str, str]] = []
    for line in text.splitlines():
        if "|" not in line:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 3:
            continue
        lower_cells = [cell.lower() for cell in cells]
        if lower_cells == ["subject", "predicate", "object"]:
            continue
        if all(re.fullmatch(r":?-+:?", cell) for cell in cells):
            continue
        parsed.append((cells[0], cells[1], cells[2]))
    return deduplicate_triplets(parsed)


def parse_output(raw_output: str, output_style: str) -> List[Tuple[str, str, str]]:
    if output_style == "json":
        return parse_triplets_json(raw_output)
    if output_style == "markdown":
        return parse_triplets_markdown(raw_output)
    raise ValueError(f"Unsupported output_style: {output_style}")


def generate_batch(llm, sampling_params, request_items: Sequence[Dict]) -> List[str]:
    prompts = [item["prompt"] for item in request_items]
    outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
    return [output.outputs[0].text for output in outputs]


def maybe_print_prompt_tokens(args: argparse.Namespace, llm, pending: Sequence[Dict]) -> None:
    if not args.debug_prompt_tokens or getattr(args, "_printed_prompt_tokens", False) or not pending:
        return

    tokenizer = llm.get_tokenizer()
    prompt = pending[0]["prompt"]
    ids = tokenizer(prompt, add_special_tokens=False, return_attention_mask=False).input_ids
    count = args.debug_prompt_token_count
    tokens = tokenizer.convert_ids_to_tokens(ids[:count])
    print(f"[Debug] First prompt token count: {len(ids)}")
    print(f"[Debug] First prompt token ids[:{count}]: {ids[:count]}")
    print(f"[Debug] First prompt tokens[:{count}]: {tokens}")
    print(f"[Debug] First prompt prefix: {prompt[: args.debug_prompt_char_count]!r}")
    args._printed_prompt_tokens = True


def flush_pending(
    args: argparse.Namespace,
    llm,
    sampling_params,
    pending: Sequence[Dict],
    completed_predictions: Dict[str, Dict],
    jsonl_path: str,
    loop,
    gpu_mem_monitor: Optional[GPUMemoryMonitor],
) -> None:
    maybe_print_prompt_tokens(args, llm, pending)
    raw_outputs = generate_batch(llm, sampling_params, pending)
    for item, raw_output in zip(pending, raw_outputs):
        key = item["key"]
        parsed_triplets = parse_output(raw_output, args.output_style)
        row = {
            "key": key,
            "index": item["idx"],
            "backend": "vllm",
            "prompt_style": "papaluca",
            "output_style": args.output_style,
            "shot_k": args.shot_k,
            "model": args.model_id_or_path or f"{args.model_type}-{args.model_size}",
            "disable_thinking": args.disable_thinking,
            "fewshot_sample_ids": item["fewshot_sample_ids"],
            "raw_output": raw_output,
            "triplets": [
                {
                    "subject": subject,
                    "predicate": predicate,
                    "object": obj,
                }
                for subject, predicate, obj in parsed_triplets
            ],
        }
        append_prediction(jsonl_path, row)
        completed_predictions[key] = row
        set_progress_postfix(loop, f"saved {key}", gpu_mem_monitor)


def save_results(args: argparse.Namespace, samples: Sequence[Dict], completed_predictions: Dict[str, Dict]) -> None:
    ordered_entries = []
    for idx, sample in enumerate(samples):
        key = get_entry_key(sample, idx)
        if key not in completed_predictions:
            continue
        row = completed_predictions[key]
        ordered_entries.append(
            {
                "sample_id": key,
                "triplets": row.get("triplets", []),
                "raw_output": row.get("raw_output", ""),
                "fewshot_sample_ids": row.get("fewshot_sample_ids", []),
            }
        )

    payload = {
        "dataset": args.dataset_name,
        "output_style": args.output_style,
        "shot_k": args.shot_k,
        "prompt_style": "papaluca",
        "model": args.model_id_or_path or f"{args.model_type}-{args.model_size}",
        "entries": ordered_entries,
    }
    output_path = Path(get_output_filename(args, "json"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[OK] saved {len(ordered_entries)} predictions to {output_path}")


def main(args: argparse.Namespace) -> None:
    from load_model import resolve_model_id

    if not Path(args.input_file).exists():
        raise FileNotFoundError(f"Missing input file: {args.input_file}")
    if args.train_path is None:
        raise ValueError("--train_path is required. Pass a local few-shot source under repair_t2d/t2d_repair_input.")
    if not args.train_path.exists():
        raise FileNotFoundError(f"Missing few-shot source: {args.train_path}")

    configure_multiprocessing()
    max_knowledge_triplets = infer_max_knowledge_triplets(args)
    model_id = resolve_model_id(args)
    llm = load_vllm(args, model_id)
    tokenizer = llm.get_tokenizer()
    sampling_params = build_sampling_params(args, tokenizer)

    print("[Info] Loading retrieval index...")
    retriever = PapalucaRetriever(
        train_path=args.train_path,
        fewshot_lang=args.fewshot_lang,
        selector_model_name=args.selector_model_name,
        selector_device=args.selector_device,
        selector_cache_dir=args.selector_cache_dir,
    )

    print("[Info] Loading samples...")
    samples = load_unified_samples(Path(args.input_file))

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    jsonl_path = get_output_filename(args, "jsonl")
    completed_predictions = load_completed_predictions(jsonl_path)
    print(f"[Info] Loaded {len(completed_predictions)} completed predictions from {jsonl_path}")

    pending: List[Dict] = []
    gpu_mem_monitor = GPUMemoryMonitor(args.show_gpu_mem)
    if args.show_gpu_mem and gpu_mem_monitor.error:
        print(f"[Warn] GPU memory monitor unavailable: {gpu_mem_monitor.error}")
    loop = tqdm(enumerate(samples), total=len(samples), desc=f"{args.dataset_name}-{args.output_style}")
    for idx, sample in loop:
        key = get_entry_key(sample, idx)
        if key in completed_predictions:
            set_progress_postfix(loop, f"resume {key}", gpu_mem_monitor)
            continue

        fewshots = retriever.retrieve(str(sample["source_text"]), args.shot_k)
        prompt = build_prompt(
            source_text=str(sample["source_text"]),
            fewshots=fewshots,
            output_style=args.output_style,
            max_knowledge_triplets=max_knowledge_triplets,
        )
        messages = [{"role": "user", "content": prompt}]
        rendered_prompt = render_prompt(args, tokenizer, messages)

        pending.append(
            {
                "idx": idx,
                "key": key,
                "fewshot_sample_ids": [example.sample_id for example in fewshots],
                "messages": messages,
                "prompt": rendered_prompt,
            }
        )

        if len(pending) >= args.batch_size:
            flush_pending(args, llm, sampling_params, pending, completed_predictions, jsonl_path, loop, gpu_mem_monitor)
            pending = []

    if pending:
        flush_pending(args, llm, sampling_params, pending, completed_predictions, jsonl_path, loop, gpu_mem_monitor)

    save_results(args, samples, completed_predictions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Papaluca-style text-to-data prompting with vLLM.")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--train_path", type=Path, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_filename", type=str, default="")
    parser.add_argument("--dataset_name", type=str, default="")
    parser.add_argument("--fewshot_lang", type=str, default="en")
    parser.add_argument("--shot_k", type=int, default=5)
    parser.add_argument("--output_style", choices=["json", "markdown"], required=True)
    parser.add_argument("--max_knowledge_triplets", type=int, default=None)
    parser.add_argument("--selector_model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--selector_device", type=str, default="cpu")
    parser.add_argument("--selector_cache_dir", type=Path, default=DEFAULT_SELECTOR_CACHE_DIR)

    parser.add_argument("--model_cache", type=str, required=True)
    parser.add_argument("--hf_token", type=str, default="")
    parser.add_argument("--model_source", type=str, default="hf")
    parser.add_argument("--model_id_or_path", type=str, default="")
    parser.add_argument("--model_size", type=str, default="8b")
    parser.add_argument("--model_type", type=str, default="llama3")
    parser.add_argument("--disable_thinking", action="store_true")
    parser.add_argument("--debug_prompt_tokens", action="store_true")
    parser.add_argument("--debug_prompt_token_count", type=int, default=16)
    parser.add_argument("--debug_prompt_char_count", type=int, default=240)

    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--quantization", type=str, default="")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--max_num_seqs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--show_gpu_mem", dest="show_gpu_mem", action="store_true")
    parser.add_argument("--hide_gpu_mem", dest="show_gpu_mem", action="store_false")
    parser.set_defaults(show_gpu_mem=True)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=1280)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    args = parser.parse_args()

    main(args)
