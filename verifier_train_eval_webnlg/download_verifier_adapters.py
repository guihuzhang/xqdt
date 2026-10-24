#!/usr/bin/env python3
"""Download published adapters into the paths used by the inference scripts."""
from __future__ import annotations

import argparse
from pathlib import Path


ADAPTER_PATHS = {
    'webnlg': {
        'Gemma3-270M': 'output_gemma3_270correct0warm/v0-20251211-213646/checkpoint-16620',
        'Gemma3-1B': 'output_gemma3_1b_correct/v4-20251211-081939/checkpoint-6648',
        'Gemma3-4B': 'output_gemma3_4b_correct/v0-20251213-035913/checkpoint-6648',
        'Gemma3-12B': 'output_gemma3_12b_correct/v0-20251215-055610/checkpoint-4432',
        'Qwen3-0.6B': 'output_qwen3_0.6correct0warm/v0-20251211-212729/checkpoint-8864',
        'Qwen3-1.7B': 'output_qwen3_1.7correct/v3-20251211-081218/checkpoint-6648',
        'Qwen3-4B': 'output_qwen3_4b_correct/v3-20251213-032130/checkpoint-4432',
        'Qwen3-8B': 'output_qwen3_8b_correct/v3-20251215-185808/checkpoint-7479',
        'Qwen3-14B': 'output_qwen3_14b_correct/v2-20251214-203429/checkpoint-6648',
        'Llama3.2-1B': 'output_llama3_1b_correct/v2-20251213-021354/checkpoint-8864',
        'Llama3.2-3B': 'output_llama3_3b_correct/v1-20251213-203738/checkpoint-5540',
        'Llama3.1-8B': 'output_llama3.1_8b_correct/v1-20251214-200900/checkpoint-5263',
    },
    'e2e19': {
        'Gemma3-270M': 'output_gemma3_270e2e19/v0-20260214-053509/checkpoint-20925',
        'Gemma3-1B': 'output_gemma3_1b_e2e19/v0-20260214-151027/checkpoint-9765',
        'Gemma3-4B': 'output_gemma3_4b_e2e19/v0-20260215-222239/checkpoint-6975',
        'Gemma3-12B': 'output_gemma3_12b_e2e/v0-20260217-040140/checkpoint-4960',
        'Qwen3-0.6B': 'output_qwen3_0.6e2e19/v0-20260214-055213/checkpoint-11160',
        'Qwen3-1.7B': 'output_qwen3_1.7e2e/v1-20260214-160705/checkpoint-15345',
        'Qwen3-4B': 'output_qwen3_4b_e2e19/v1-20260216-031110/checkpoint-8370',
        'Qwen3-8B': 'output_qwen3_8b_e2e/v0-20260216-045117/checkpoint-8680',
        'Qwen3-14B': 'output_qwen3_14b_e2e/v0-20260217-133629/checkpoint-4960',
        'Llama3.2-1B': 'output_llama3_1b_e2e19/v0-20260214-154428/checkpoint-9765',
        'Llama3.2-3B': 'output_llama3_3b_e2e19/v0-20260214-225224/checkpoint-6975',
        'Llama3.1-8B': 'output_llama3.1_8b_e2e/v1-20260216-051935/checkpoint-8680',
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['webnlg', 'e2e19'], required=True)
    parser.add_argument('--model', help='Model name, e.g. Qwen3-4B; omit to download all 12.')
    parser.add_argument('--list', action='store_true', help='List downloads without accessing the network.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    entries = list(ADAPTER_PATHS[args.dataset].items())
    if args.model:
        entries = [(model, path) for model, path in entries if model == args.model]
        if not entries:
            raise ValueError(f'Unknown model: {args.model}')
    for model, path in entries:
        dataset = 'e2e' if args.dataset == 'e2e19' else 'webnlg'
        repo_id = f'Loria-MosAIk/xqdt-{dataset}-{model.lower()}'
        destination = root / 'checkpoints' / ('checkpoints_' + args.dataset) / path
        print(f'{repo_id} -> {destination.relative_to(root)}', flush=True)
        if not args.list:
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id, local_dir=destination,
                              allow_patterns=['*.json', '*.safetensors', '*.model', '*.txt', '*.jinja'])


if __name__ == '__main__':
    main()
