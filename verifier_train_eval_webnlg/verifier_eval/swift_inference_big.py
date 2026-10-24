#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference script for Triple Error Detection
Using Swift PtEngine (official method)
"""
import argparse
import json
import os
from pathlib import Path
import sys
from typing import List, Dict
import torch

from swift.llm import (
    PtEngine, RequestConfig, get_model_tokenizer, get_template, InferRequest
)
from swift.tuners import Swift
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from verifier_train_eval_webnlg.path_config import (
    CHECKPOINTS_WEBNLG_DIR,
    ERROR_PREDICTION_WEBNLG_DIR,
    XQDT_OUTPUTS_DIR,
)


def format_triple(triple_str: str) -> str:
    """Format triple to [S] subject [P] predicate [O] object"""
    parts = [p.strip() for p in triple_str.split('|')]
    if len(parts) == 3:
        return f"[S] {parts[0]} [P] {parts[1]} [O] {parts[2]}"
    return triple_str


def build_query(text: str, triples: List[str]) -> str:
    """Build query prompt"""
    triples_formatted = '\n'.join(
        f'{i}. {format_triple(t)}' for i, t in enumerate(triples, start=1))

    query = f"""Verify if the triples align with the text. Find missing, extra, or incorrect triples.
TEXT: {text}
TRIPLES:
{triples_formatted}
Output as markdown table with Type and Triple columns."""
    return query


def parse_prediction(response: str) -> Dict[str, List[str]]:
    """Parse model prediction from markdown table"""
    result = {'missing': [], 'extra': [], 'incorrect': []}

    if not response or 'All correct' in response:
        return result

    lines = response.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line or '|' not in line:
            continue
        if 'Type' in line or '----' in line:
            continue

        parts = [p.strip() for p in line.split('|')]
        if len(parts) < 3:
            continue

        error_type = parts[1].strip().lower()
        triple = parts[2].strip()

        if 'missing' in error_type:
            result['missing'].append(triple)
        elif 'extra' in error_type:
            result['extra'].append(triple)
        elif 'incorrect' in error_type:
            result['incorrect'].append(triple)

    return result


def run_inference(
        base_model: str,
        lora_checkpoint: str,
        dataset_json: str,
        output_file: str,
        template_type: str = None,
        default_system: str = None,
        max_samples: int = None,
        batch_size: int = 1
):
    """
    Load model once, run inference on all samples, save results

    Args:
        base_model: Base model name or path (e.g., 'Qwen/Qwen3-0.6B')
        lora_checkpoint: LoRA checkpoint directory path
        dataset_json: Path to test/val JSON
        output_file: Output JSON path
        template_type: Template type (None = auto detect)
        default_system: System prompt for Qwen models
        max_samples: Max samples to process
        batch_size: Batch size for inference
    """
    print(f"\nInference Setup")
    print(f"Base model: {base_model}")
    print(f"LoRA checkpoint: {lora_checkpoint}")
    print(f"Dataset: {dataset_json}")
    print(f"Output: {output_file}")
    if default_system:
        print(f"System: {default_system}")

    # Load base model and tokenizer
    if not Path(dataset_json).is_file():
        raise FileNotFoundError(f"Required test data not found: {dataset_json}")
    print(f"\nLoading model...")
    torch.set_grad_enabled(False)

    model, tokenizer = get_model_tokenizer(
        base_model,
        model_kwargs = {
            'device_map': 'auto',
            'torch_dtype': torch.bfloat16,
            # 'attn_implementation': 'flash_attention_2',
        }
    )
    # lora_checkpoint = str(Path(lora_checkpoint).resolve())  # Convert to absolute path

    # Load LoRA weights - model_id is the checkpoint directory
    model = Swift.from_pretrained(
        model,  # The base model
        model_id=lora_checkpoint,  # model_id: local checkpoint directory
        adapter_name='default'  # Optional adapter name
    )
    model.eval()
    # Get template
    template_type = template_type or 'AUTO'
    template = get_template(template_type, tokenizer, default_system=default_system)

    # Create inference engine
    engine = PtEngine.from_model_template(model, template, max_batch_size=batch_size)
    request_config = RequestConfig(max_tokens=1024, temperature=0.3)

    print("Model loaded!\n")

    # Load dataset
    print(f"Loading dataset...")
    with open(dataset_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if max_samples:
        data = data[:max_samples]

    print(f"Total samples: {len(data)}\n")

    # Run inference
    print("Running inference...")
    results = []

    for i in tqdm(range(0, len(data), batch_size), desc="Processing"):
        batch = data[i:i + batch_size]

        try:
            # Prepare batch requests
            infer_requests = []
            for sample in batch:
                text = sample.get('input_text', '')
                triples = sample.get('triples', [])
                query = build_query(text, triples)

                infer_requests.append(
                    InferRequest(messages=[{'role': 'user', 'content': query}])
                )

            # Run batch inference
            resp_list = engine.infer(infer_requests, request_config, use_tqdm=False)
            if len(resp_list) != len(batch):
                raise RuntimeError(f"Expected {len(batch)} responses, got {len(resp_list)}")

            # Process responses
            for j, (sample, resp) in enumerate(zip(batch, resp_list)):
                response = resp.choices[0].message.content
                if not isinstance(response, str) or not response.strip():
                    raise RuntimeError(f"Empty model response at sample {i + j}")
                prediction = parse_prediction(response)

                results.append({
                    'sample_id': i + j,
                    'input_text': sample.get('input_text', ''),
                    'input_triples': sample.get('triples', []),
                    'label': sample.get('label', ''),
                    'gt_extra_triples': sample.get('extra_contents', []),
                    'gt_missing_triples': sample.get('missing_triples', []),
                    'gt_incorrect_mappings': sample.get('incorrect_mappings', []),
                    'prompt': infer_requests[j].messages[0]['content'],
                    'model_output': response,
                    'parsed_output': prediction
                })

        except Exception as e:
            print(f"\nError on batch starting at sample {i}: {e}")
            import traceback
            traceback.print_exc()
            raise

    # Save results
    print(f"\nSaving to {output_file}...")
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nCompleted!")
    print(f"Processed: {len(results)} samples")
    print(f"Saved to: {output_file}\n")

    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['Qwen3-8B', 'Llama3.1-8B', 'Qwen3-14B'])
    parser.add_argument('--max-samples', type=int)
    parser.add_argument('--output-dir', type=Path, default=XQDT_OUTPUTS_DIR)
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples < 1:
        parser.error('--max-samples must be positive')
    # Set GPU
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
    dataset_json = str(ERROR_PREDICTION_WEBNLG_DIR / 'test_all_samples_corrected.json')
    output_dir = args.output_dir
    checkpoint_root = CHECKPOINTS_WEBNLG_DIR

    # Qwen3 8B
    if args.model is None or args.model == 'Qwen3-8B':
        run_inference(
            base_model='Qwen/Qwen3-8B',
            lora_checkpoint=str(checkpoint_root / 'output_qwen3_8b_correct' / 'v3-20251215-185808' / 'checkpoint-7479'),
            dataset_json=dataset_json,
            output_file=str(output_dir / 'test_predictions_qwen3_8b_7479.json'),
            template_type='qwen3',
            default_system="Identify extra, missing and incorrect triples precisely.",
            max_samples=args.max_samples,
            batch_size=2
        )
    # Llama-3.1-8B
    if args.model is None or args.model == 'Llama3.1-8B':
        run_inference(
            base_model='LLM-Research/Meta-Llama-3.1-8B-Instruct',
            lora_checkpoint=str(checkpoint_root / 'output_llama3.1_8b_correct' / 'v1-20251214-200900' / 'checkpoint-5263'),
            dataset_json=dataset_json,
            output_file=str(output_dir / 'test_predictions_llama3_1_8b_5263.json'),
            default_system="Identify extra, missing and incorrect triples precisely.",
            template_type='llama3_2',
            max_samples=args.max_samples,
            batch_size=2
        )
    if args.model is None or args.model == 'Qwen3-14B':
        run_inference(
            base_model='Qwen/Qwen3-14B',
            lora_checkpoint=str(checkpoint_root / 'output_qwen3_14b_correct' / 'v2-20251214-203429' / 'checkpoint-6648'),
            dataset_json=dataset_json,
            output_file=str(output_dir / 'test_predictions_qwen3_14b_6648.json'),
            template_type='qwen3',
            default_system="Identify extra, missing and incorrect triples precisely.",
            max_samples=args.max_samples,
            batch_size=2
        )
