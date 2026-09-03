#!/usr/bin/env python3
"""Shared prompting and inference utilities for human-alignment experiments."""

from __future__ import annotations

import gc
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from tqdm import tqdm


PROMPT = """Verify if the data has missing, extra, or incorrect contents, compared with the text.

TEXT: {text}

DATA:
{triples}

Find three error types:
1. Extra: content expressed in the text but unsupported by the data.
2. Missing: a data unit containing two or three elements not expressed in the text.
3. Incorrect: one element is wrong, or the subject and object are reversed.

Return only a markdown table with Type and Triple columns. Use one [S] subject [P] predicate [O] object string per row. If there is no error, output "All correct"."""


def format_triple(triple: Any) -> str:
    if isinstance(triple, dict):
        subject = triple.get("subject", "")
        predicate = triple.get("property", triple.get("predicate", ""))
        obj = triple.get("object", "")
        return f"[S] {subject} [P] {predicate} [O] {obj}"
    value = str(triple)
    parts = [part.strip() for part in re.split(r"\s*\|\s*", value)]
    if len(parts) == 3:
        return f"[S] {parts[0]} [P] {parts[1]} [O] {parts[2]}"
    return value


def build_prompt(text: str, triples: Iterable[Any]) -> str:
    formatted = "\n".join(
        f"{index}. {format_triple(triple)}"
        for index, triple in enumerate(triples, start=1)
    )
    return PROMPT.format(text=text, triples=formatted)


def parse_response(response: str) -> Dict[str, List[str]]:
    parsed = {"missing": [], "extra": [], "incorrect": []}
    if not response or "all correct" in response.lower():
        return parsed
    for raw_line in response.splitlines():
        line = raw_line.strip()
        if not line.startswith("|") or "---" in line:
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < 2 or cells[0].lower() == "type":
            continue
        error_type, triple = cells[0].lower(), " | ".join(cells[1:]).strip()
        if triple.lower() in {"", "-", "none", "n/a", "null"}:
            continue
        if "missing" in error_type or "omission" in error_type:
            parsed["missing"].append(triple)
        elif "extra" in error_type or "additional" in error_type:
            parsed["extra"].append(triple)
        elif "incorrect" in error_type or "wrong" in error_type:
            parsed["incorrect"].append(triple)
    return parsed


def load_jsonl(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    records: Dict[str, Dict[str, Any]] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                record = json.loads(line)
                records[str(record["sample_id"])] = record
    return records


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


class OpenRouterBackend:
    def __init__(self, model: str, max_tokens: int, retries: int) -> None:
        from openai import OpenAI

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.retries = retries
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self.cost_reported = False

    def infer(self, prompts: List[str]) -> List[str]:
        outputs = []
        for prompt in tqdm(prompts, desc="OpenRouter", unit="sample", leave=False):
            for attempt in range(self.retries + 1):
                try:
                    completion = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=self.max_tokens,
                        seed=2023,
                    )
                    outputs.append(completion.choices[0].message.content or "")
                    usage = completion.usage
                    if usage is not None:
                        usage_data = (
                            usage.model_dump()
                            if hasattr(usage, "model_dump")
                            else vars(usage)
                        )
                        self.prompt_tokens += int(usage_data.get("prompt_tokens") or 0)
                        self.completion_tokens += int(usage_data.get("completion_tokens") or 0)
                        self.total_tokens += int(usage_data.get("total_tokens") or 0)
                        cost = usage_data.get("cost")
                        if cost is not None:
                            self.total_cost += float(cost)
                            self.cost_reported = True
                    break
                except Exception:
                    if attempt == self.retries:
                        raise
                    time.sleep(2 ** attempt)
        return outputs

    def close(self) -> None:
        print("OpenRouter usage:")
        print(f"  prompt_tokens:     {self.prompt_tokens}")
        print(f"  completion_tokens: {self.completion_tokens}")
        print(f"  total_tokens:      {self.total_tokens}")
        if self.cost_reported:
            print(f"  cost_usd:          ${self.total_cost:.6f}")
        else:
            print("  cost_usd:          unavailable in API response")


class SwiftVllmBackend:
    def __init__(
        self,
        model: str,
        max_tokens: int,
        batch_size: int,
        max_model_len: int,
        gpu_memory_utilization: float,
        tensor_parallel_size: int,
        max_num_seqs: int,
        model_type: Optional[str],
        template_type: Optional[str],
        dtype: str,
        disable_thinking: bool,
    ) -> None:
        import torch

        os.environ.setdefault("USE_HF", "1")
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        try:
            from swift.llm import InferRequest, RequestConfig, VllmEngine
        except (ImportError, ModuleNotFoundError):
            from swift import InferRequest, RequestConfig, VllmEngine

        self.InferRequest = InferRequest
        self.request_config = RequestConfig(
            max_tokens=max_tokens, temperature=0.0, seed=2023
        )
        self.batch_size = batch_size
        self.disable_thinking = disable_thinking
        engine_kwargs = dict(
            use_hf=True,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            max_num_seqs=max_num_seqs,
            enforce_eager=True,
            seed=2023,
        )
        if dtype == "bfloat16":
            engine_kwargs["torch_dtype"] = torch.bfloat16
        if model_type:
            engine_kwargs["model_type"] = model_type
        if template_type:
            engine_kwargs["template_type"] = template_type
        self.engine = VllmEngine(model, **engine_kwargs)

    def infer(self, prompts: List[str]) -> List[str]:
        outputs: List[str] = []
        for start in tqdm(
            range(0, len(prompts), self.batch_size),
            desc="vLLM",
            unit="batch",
            leave=False,
        ):
            requests = [
                self.InferRequest(messages=[{
                    "role": "user",
                    "content": f"{prompt}\n\n/no_think" if self.disable_thinking else prompt,
                }])
                for prompt in prompts[start : start + self.batch_size]
            ]
            responses = self.engine.infer(
                requests, self.request_config, use_tqdm=False
            )
            outputs.extend(
                response.choices[0].message.content or "" for response in responses
            )
        return outputs

    def close(self) -> None:
        del self.engine
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except (ImportError, RuntimeError):
            pass


def make_backend(args):
    if args.backend == "openrouter":
        return OpenRouterBackend(args.model, args.max_tokens, args.retries)
    return SwiftVllmBackend(
        args.model,
        args.max_tokens,
        args.batch_size,
        args.max_model_len,
        args.gpu_memory_utilization,
        args.tensor_parallel_size,
        args.max_num_seqs,
        args.model_type,
        args.template,
        args.dtype,
        args.disable_thinking,
    )


def add_backend_args(parser) -> None:
    parser.add_argument("--backend", choices=["openrouter", "vllm"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--model-type", default=None)
    parser.add_argument("--template", default=None)
    parser.add_argument("--dtype", choices=["auto", "bfloat16"], default="bfloat16")
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Append /no_think to each prompt (for Qwen3 hybrid-thinking models).",
    )
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")


def model_slug(args) -> str:
    value = args.model_name or args.model.rsplit("/", 1)[-1]
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)
