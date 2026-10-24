#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run BT5-style reference-less D2T evaluation with Swift's vLLM backend.

This is intentionally close to the old dt-explainable-metric evaluator:
- same prompt text
- same model/evaluator checkpoint names
- same parsed_output and metrics schema

The main difference is the inference backend: Swift VllmEngine instead of PtEngine.
"""

from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPAIR_ROOT = SCRIPT_DIR
REPO_ROOT = REPAIR_ROOT.parent
DEFAULT_OUTPUT_DIR = REPAIR_ROOT / "eval" / "eval_vllm"
DEFAULT_SYSTEM = "Identify extra, missing and incorrect triples precisely."

MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "gemma4b": {
        "name": "gemma3_4b_step_6648",
        "base_model": "google/gemma-3-4b-it",
        "lora_checkpoint": "checkpoints/checkpoints_webnlg/output_gemma3_4b_correct/v0-20251213-035913/checkpoint-6648",
        "template_type": "gemma3_text",
        "default_system": None,
        "use_hf": "1",
        "batch_size": 1,
    },
    "llama8b": {
        "name": "llama3_1_8b_step_5263",
        "base_model": "meta-llama/Llama-3.1-8B-Instruct",
        "lora_checkpoint": "checkpoints/checkpoints_webnlg/output_llama3.1_8b_correct/v1-20251214-200900/checkpoint-5263",
        "template_type": "llama3_2",
        "default_system": DEFAULT_SYSTEM,
        "use_hf": "1",
        "batch_size": 1,
    },
    "qwen8b": {
        "name": "qwen3_8b_step_7479",
        "base_model": "Qwen/Qwen3-8B",
        "lora_checkpoint": "checkpoints/checkpoints_webnlg/output_qwen3_8b_correct/v3-20251215-185808/checkpoint-7479",
        "template_type": "qwen3",
        "default_system": DEFAULT_SYSTEM,
        "use_hf": "1",
        "batch_size": 1,
    },
}


def import_swift_api() -> Dict[str, Any]:
    """Support both ms-swift 3.x (`swift.llm`) and 4.x top-level `swift` APIs."""
    try:
        from swift.llm import InferRequest, RequestConfig, VllmEngine, get_template

        return {
            "InferRequest": InferRequest,
            "RequestConfig": RequestConfig,
            "VllmEngine": VllmEngine,
            "get_template": get_template,
            "legacy_llm_api": True,
        }
    except ModuleNotFoundError as exc:
        if exc.name not in {"swift.llm", "swift"}:
            raise
    except ImportError:
        pass

    from swift import InferRequest, RequestConfig, VllmEngine

    try:
        from swift import get_template
    except ImportError:
        get_template = None

    return {
        "InferRequest": InferRequest,
        "RequestConfig": RequestConfig,
        "VllmEngine": VllmEngine,
        "get_template": get_template,
        "legacy_llm_api": False,
    }


def make_infer_request(infer_request_cls, query: str, system: Optional[str], *, pass_system: bool):
    messages = [{"role": "user", "content": query}]
    if not pass_system or not system:
        return infer_request_cls(messages=messages)
    try:
        return infer_request_cls(messages=messages, system=system)
    except TypeError:
        return infer_request_cls(messages=[{"role": "system", "content": system}, *messages])


def configure_multiprocessing() -> None:
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass


def selected_model_keys(raw_models: List[str]) -> List[str]:
    if "all" in raw_models:
        return list(MODEL_CONFIGS.keys())
    ordered = []
    for key in raw_models:
        if key not in ordered:
            ordered.append(key)
    return ordered


def ensure_file(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")
    return path


def resolve_local_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def clean_stem(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_eval_input"):
        stem = stem[: -len("_eval_input")]
    return stem


def temp_tag(value: float) -> str:
    return str(value).replace(".", "p")


def parse_optional_bool(value: str, default: str) -> bool:
    raw = default if value == "auto" else value
    return str(raw).strip().lower() in {"1", "true", "yes", "y"}


def set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    os.environ["PYTHONHASHSEED"] = str(seed)


def format_triple(triple_str: str) -> str:
    parts = [part.strip() for part in triple_str.split("|")]
    if len(parts) == 3:
        return f"[S] {parts[0]} [P] {parts[1]} [O] {parts[2]}"
    return triple_str


def build_query(text: str, triples: List[str]) -> str:
    triples_formatted = "\n".join(
        f"{idx}. {format_triple(triple)}" for idx, triple in enumerate(triples, start=1)
    )
    return f"""Verify if the triples align with the text. Find missing, extra, or incorrect triples.
TEXT: {text}
TRIPLES:
{triples_formatted}
Output as markdown table with Type and Triple columns."""


def parse_prediction(response: str) -> Dict[str, List[str]]:
    result = {"missing": [], "extra": [], "incorrect": []}
    if not response or "all correct" in response.lower():
        return result

    for raw_line in response.strip().splitlines():
        line = raw_line.strip()
        if not line or "|" not in line:
            continue
        if "Type" in line or "----" in line:
            continue

        parts = [part.strip() for part in line.split("|")]
        if len(parts) < 3:
            continue

        error_type = parts[1].lower()
        triple = parts[2].strip()
        if triple.lower() in {"", "-", "—", "n/a", "none", "null"}:
            continue
        if "missing" in error_type:
            result["missing"].append(triple)
        elif "extra" in error_type:
            result["extra"].append(triple)
        elif "incorrect" in error_type:
            result["incorrect"].append(triple)

    return result


def unique_count(items: Iterable[str]) -> int:
    normalized = {
        re.sub(r"\s+", " ", str(item)).strip().lower()
        for item in items
        if str(item).strip()
    }
    return len(normalized)


def compute_prf(parsed_output: Dict[str, List[str]], num_source_triples: int) -> Dict[str, float]:
    num_missing = unique_count(parsed_output.get("missing", []))
    num_extra = unique_count(parsed_output.get("extra", []))
    num_incorrect = unique_count(parsed_output.get("incorrect", []))

    tp = max(0, int(num_source_triples) - num_missing - num_incorrect)
    fp = num_extra + num_incorrect
    fn = num_missing + num_incorrect

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "num_source_triples": int(num_source_triples),
        "num_missing": int(num_missing),
        "num_extra": int(num_extra),
        "num_incorrect": int(num_incorrect),
    }


def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_tp = sum(item["metrics"]["TP"] for item in results)
    total_fp = sum(item["metrics"]["FP"] for item in results)
    total_fn = sum(item["metrics"]["FN"] for item in results)

    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    n = len(results)
    macro_p = sum(item["metrics"]["precision"] for item in results) / n if n else 0.0
    macro_r = sum(item["metrics"]["recall"] for item in results) / n if n else 0.0
    macro_f1 = sum(item["metrics"]["f1"] for item in results) / n if n else 0.0

    return {
        "num_samples": int(n),
        "micro": {
            "precision": float(micro_p),
            "recall": float(micro_r),
            "f1": float(micro_f1),
            "TP": int(total_tp),
            "FP": int(total_fp),
            "FN": int(total_fn),
        },
        "macro": {
            "precision": float(macro_p),
            "recall": float(macro_r),
            "f1": float(macro_f1),
        },
    }


def clear_cuda_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def parse_torch_dtype(dtype: str):
    if dtype == "auto":
        return None
    import torch

    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return mapping[dtype]


def build_request_config(args: argparse.Namespace, temperature: float):
    request_config_cls = import_swift_api()["RequestConfig"]

    return request_config_cls(
        max_tokens=args.max_tokens,
        temperature=temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
    )


def load_vllm_engine(args: argparse.Namespace, config: Dict[str, Any]):
    swift_api = import_swift_api()
    vllm_engine_cls = swift_api["VllmEngine"]
    get_template = swift_api["get_template"]
    legacy_llm_api = bool(swift_api["legacy_llm_api"])

    lora_checkpoint = resolve_local_path(config["lora_checkpoint"])
    if not lora_checkpoint.is_dir():
        raise FileNotFoundError(f"LoRA checkpoint not found: {lora_checkpoint}")

    previous_use_hf = os.environ.get("USE_HF")
    use_hf = parse_optional_bool(args.use_hf, str(config.get("use_hf", "0")))
    os.environ["USE_HF"] = "1" if use_hf else "0"

    cache_dir = (REPAIR_ROOT / args.model_cache / "vllm").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    engine_kwargs: Dict[str, Any] = {"download_dir": str(cache_dir)}

    try:
        common_kwargs = dict(
            torch_dtype=parse_torch_dtype(args.dtype),
            adapters=[str(lora_checkpoint)],
            use_hf=use_hf,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs or 256,
            quantization=args.quantization or None,
            enforce_eager=args.enforce_eager,
            seed=args.seed,
            max_lora_rank=args.max_lora_rank,
            engine_kwargs=engine_kwargs,
        )
        if legacy_llm_api:
            engine = vllm_engine_cls(str(config["base_model"]), **common_kwargs)
            if get_template is None:
                raise ImportError("Cannot import get_template from swift.llm")
            template = get_template(
                str(config["template_type"]),
                engine.tokenizer,
                default_system=config.get("default_system"),
            )
        else:
            engine = vllm_engine_cls(
                str(config["base_model"]),
                template_type=str(config["template_type"]),
                **common_kwargs,
            )
            template = None
        return engine, template, previous_use_hf
    except Exception:
        if previous_use_hf is None:
            os.environ.pop("USE_HF", None)
        else:
            os.environ["USE_HF"] = previous_use_hf
        raise


def restore_use_hf(previous_use_hf: Optional[str]) -> None:
    if previous_use_hf is None:
        os.environ.pop("USE_HF", None)
    else:
        os.environ["USE_HF"] = previous_use_hf


def run_one(
    *,
    args: argparse.Namespace,
    model_key: str,
    config: Dict[str, Any],
    data: List[Dict[str, Any]],
    input_json: Path,
    temperature: float,
) -> Dict[str, Any]:
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    input_stem = clean_stem(input_json)
    run_tag = args.run_tag or f"vllm_temp{temp_tag(temperature)}_seed{args.seed}"
    output_path = output_dir / f"{input_stem}_{config['name']}_generation_ref_less_{run_tag}_eval.json"

    if output_path.exists() and not args.overwrite:
        print(f"Skipping existing output: {output_path}")
        results = json.loads(output_path.read_text(encoding="utf-8"))
        summary = summarize_results(results)
        summary.update({"output_file": str(output_path), "model_key": model_key, "temperature": temperature})
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary

    print("\nInference Setup")
    print(f"Input: {input_json}")
    print(f"Model key: {model_key}")
    print(f"Base model: {config['base_model']}")
    print(f"LoRA checkpoint: {resolve_local_path(config['lora_checkpoint'])}")
    print(f"Template: {config['template_type']}")
    print(f"Temperature: {temperature}")
    print(f"Seed: {args.seed}")
    print(f"Output: {output_path}")

    set_seed(args.seed)
    engine = None
    template = None
    previous_use_hf = None
    results: List[Dict[str, Any]] = []

    try:
        engine, template, previous_use_hf = load_vllm_engine(args, config)
        infer_request_cls = import_swift_api()["InferRequest"]

        request_config = build_request_config(args, temperature)
        batch_size = int(args.batch_size or config["batch_size"])
        for start in tqdm(range(0, len(data), batch_size), desc=f"{model_key} t={temperature}"):
            batch = data[start:start + batch_size]
            infer_requests = []
            prompts = []
            for sample in batch:
                source_triples = list(sample.get("source_triples") or sample.get("normalized_generated_triples", []))
                query = build_query(str(sample.get("input_text", "")), source_triples)
                prompts.append(query)
                infer_requests.append(
                    make_infer_request(
                        infer_request_cls,
                        query,
                        config.get("default_system"),
                        pass_system=template is None,
                    )
                )

            infer_kwargs = {"use_tqdm": False}
            if template is not None:
                infer_kwargs["template"] = template
            resp_list = engine.infer(infer_requests, request_config, **infer_kwargs)

            for offset, (sample, prompt, resp) in enumerate(zip(batch, prompts, resp_list)):
                response = resp.choices[0].message.content
                parsed = parse_prediction(response)
                source_triples = list(sample.get("source_triples") or sample.get("normalized_generated_triples", []))
                metrics = compute_prf(parsed, len(source_triples))
                results.append(
                    {
                        "sample_id": sample.get("sample_id", start + offset),
                        "dataset_name": sample.get("dataset_name", ""),
                        "baseline_name": sample.get("baseline_name", ""),
                        "input_text": sample.get("input_text", ""),
                        "reference_texts": sample.get("reference_texts", []),
                        "source_triples": source_triples,
                        "raw_source_triples": sample.get("raw_source_triples", []),
                        "prompt": prompt,
                        "model_output": response,
                        "parsed_output": parsed,
                        "metrics": metrics,
                    }
                )

        output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        summary = summarize_results(results)
        summary.update(
            {
                "input_json": str(input_json),
                "output_file": str(output_path),
                "model_key": model_key,
                "model_name": config["name"],
                "base_model": config["base_model"],
                "lora_checkpoint": str(resolve_local_path(config["lora_checkpoint"])),
                "backend": "swift_vllm",
                "temperature": temperature,
                "seed": args.seed,
                "batch_size": batch_size,
                "max_tokens": args.max_tokens,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "repetition_penalty": args.repetition_penalty,
            }
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary
    finally:
        restore_use_hf(previous_use_hf)
        del engine, template
        clear_cuda_memory()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ref-less D2T evaluator with Swift VllmEngine.")
    parser.add_argument("--input_json", type=Path, nargs="+", required=True)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--models", nargs="+", default=["qwen8b"], choices=["all", *MODEL_CONFIGS.keys()])
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--temperatures", type=float, nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_tag", type=str, default="")
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--max_num_seqs", type=int, default=None)
    parser.add_argument("--quantization", type=str, default="")
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--max_lora_rank", type=int, default=16)
    parser.add_argument("--model_cache", type=str, default=".cache")
    parser.add_argument("--use_hf", choices=["auto", "0", "1"], default="auto")

    # Defaults are None to mimic the old Swift PtEngine evaluator, which only set max_tokens and temperature.
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--repetition_penalty", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_multiprocessing()
    os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    input_jsons = [ensure_file(path) for path in args.input_json]
    model_keys = selected_model_keys(args.models)
    temperatures = args.temperatures if args.temperatures is not None else [args.temperature]
    summaries = []

    for input_json in input_jsons:
        data = json.loads(input_json.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"Expected list input in {input_json}")
        if args.max_samples is not None:
            data = data[: args.max_samples]
        for model_key in model_keys:
            for temperature in temperatures:
                summaries.append(
                    run_one(
                        args=args,
                        model_key=model_key,
                        config=MODEL_CONFIGS[model_key],
                        data=data,
                        input_json=input_json,
                        temperature=temperature,
                    )
                )

    print("\nCombined summary:")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
