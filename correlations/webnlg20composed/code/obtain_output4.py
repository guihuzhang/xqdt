#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Team Generations Evaluation - Batch Processing Version
Evaluates multiple published adapters on WebNLG 2020
"""
import argparse
import json
import os
import re
from pathlib import Path
from typing import List, Dict
import random
import numpy as np
import torch
from tqdm import tqdm

from swift.llm import (
    PtEngine, RequestConfig, get_model_tokenizer, get_template, InferRequest
)
from swift.tuners import Swift
from huggingface_hub import snapshot_download

BASE_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = BASE_DIR.parent
DEFAULT_TEAM_DIR = PACKAGE_DIR / 'webnlg2020_inputs' / 'team-generations'
DEFAULT_TRIPLES_FILE = PACKAGE_DIR / 'webnlg2020_inputs' / 'triples_modified.txt'
DEFAULT_OUTPUT_DIR = PACKAGE_DIR / 'xqdt_results'

# Datasets to evaluate


CHECKPOINTS = [
    {
        'base_model': 'Qwen/Qwen3-4B',
        'lora_checkpoint': 'Loria-MosAIk/xqdt-webnlg-qwen3-4b',
        'template_type': 'qwen3',
        'system_prompt': "Identify extra, missing and incorrect triples precisely.",
        'batch_size': 2,
        'name': 'qwen3_4b_4432',
    },
    {
        'base_model': 'meta-llama/Llama-3.2-3B-Instruct',
        'lora_checkpoint': 'Loria-MosAIk/xqdt-webnlg-llama3.2-3b',
        'template_type': 'llama3_2',
        'system_prompt': "Identify extra, missing and incorrect triples precisely.",
        'batch_size': 4,
        'name': 'llama3.2_3b_5540',
    },
    {
        'base_model': 'google/gemma-3-4b-it',
        'lora_checkpoint': 'Loria-MosAIk/xqdt-webnlg-gemma3-4b',
        'template_type': 'gemma3_text',
        'system_prompt': None,
        'batch_size': 4,
        'name': 'gemma3_4b_6648',
    },
]


# Global settings
MAX_TOKENS = 1024
TEMPERATURE = 0.3
RANDOM_SEED = 2023
MISSING_ENTRY_ID = 106


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--team-dir', type=Path, default=DEFAULT_TEAM_DIR)
    parser.add_argument('--triples-file', type=Path, default=DEFAULT_TRIPLES_FILE)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--skip-existing', action='store_true')
    return parser.parse_args()


def require_input_path(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f'{label} not found: {path}. '
            f'Place WebNLG20 inputs under {PACKAGE_DIR / "webnlg2020_inputs"} '
            f'or pass --{label.replace("_", "-")} explicitly.'
        )
    return path


def build_dataset_configs(args: argparse.Namespace) -> List[Dict]:
    team_dir = require_input_path(args.team_dir, 'team_dir')
    triples_file = require_input_path(args.triples_file, 'triples_file')
    return [{
        'name': 'webnlg2020',
        'team_dir': str(team_dir),
        'triples_file': str(triples_file),
        'output_dir': str(args.output_dir),
    }]


def set_random_seed(seed: int):
    """Set random seed for reproducibility across all libraries"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def clean_entity(entity: str) -> str:
    entity = re.sub(r'@[a-z]{2,3}$', '', entity)
    entity = re.sub(r'\^\^xsd:\w+', '', entity)
    entity = re.sub(r'\^\^<http://[^>]+>', '', entity)
    entity = entity.strip('"\'')
    entity = entity.replace('_', ' ')
    return entity.strip()


def clean_predicate(predicate: str) -> str:
    predicate = predicate.replace('/', ' ')
    predicate = re.sub(r'([a-z])([A-Z])', r'\1 \2', predicate)
    predicate = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', predicate)
    return predicate.strip().lower()


def parse_triple_line(line: str) -> List[str]:
    triples = line.split('<br>')
    cleaned_triples = []
    for triple in triples:
        triple = triple.strip()
        if triple and '|' in triple:
            cleaned_triples.append(triple)
    return cleaned_triples


def format_triple_to_prompt(tmp_triple: str) -> str:
    parts = [p.strip() for p in tmp_triple.split('|')]
    if len(parts) != 3:
        return tmp_triple
    subject, predicate, obj = parts
    subject = clean_entity(subject)
    predicate = clean_predicate(predicate)
    obj = clean_entity(obj)
    return f"[S] {subject} [P] {predicate} [O] {obj}"


def build_evaluation_query(text: str, triples: List[str]) -> str:
    triples_formatted = '\n'.join(
        f'{i}. {format_triple_to_prompt(t)}'
        for i, t in enumerate(triples, start=1))
    query = f"""Verify if the triples align with the text. Find missing, extra, or incorrect triples.
TEXT: {text}
TRIPLES:
{triples_formatted}
Output as markdown table with Type and Triple columns."""
    return query


def parse_model_response(response: str) -> Dict[str, List[str]]:
    result = {'missing': [], 'extra': [], 'incorrect': []}
    if not isinstance(response, str) or not response.strip():
        raise ValueError("Empty model response")
    response = re.sub(r'<think>\s*</think>', '', response).strip()
    for line in response.strip().splitlines():
        if '|' not in line:
            continue
        parts = [part.strip() for part in line.strip().strip('|').split('|')]
        if len(parts) < 2:
            continue
        lowered = [part.strip('* ').lower() for part in parts]
        if lowered[0] == 'type' and lowered[1].startswith('triple'):
            continue
        if all(re.fullmatch(r':?[-=]{3,}:?', part) for part in parts):
            continue
        error_type = lowered[0]
        triple = ' '.join(parts[1:]).strip()
        labels = {
            'missing': 'missing', 'omission': 'missing', 'omitted': 'missing',
            'extra': 'extra', 'additional': 'extra',
            'incorrect': 'incorrect', 'wrong': 'incorrect', 'error': 'incorrect',
        }
        if error_type not in labels:
            raise ValueError(f"Unrecognized error-table row: {line}")
        if not triple:
            raise ValueError(f"Missing triple in error-table row: {line}")
        result[labels[error_type]].append(triple)
    if any(result.values()):
        return result
    positive = re.fullmatch(
        r'(?:all (?:triples (?:are )?)?correct|no errors?(?: (?:found|detected))?|'
        r'(?:the triples (?:and (?:the )?text )?are )?fully aligned)[.!]?',
        response.strip().strip('* ').lower(),
    )
    if positive:
        return result
    raise ValueError("Response contains neither explicit all-correct text nor error rows")


def validate_results(results: List[Dict], dataset: List[Dict]):
    expected = {(entry['team'], entry['entry_id']) for entry in dataset}
    if not dataset or len(expected) != len(dataset):
        raise ValueError("Evaluation dataset is empty or contains duplicate keys")
    if not isinstance(results, list):
        raise ValueError("Expected a result list")
    actual = [(entry['team'], entry['entry_id']) for entry in results]
    if len(actual) != len(dataset) or len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError(
            f"Incomplete or mismatched output: expected {len(dataset)} unique rows, "
            f"received {len(actual)} rows and {len(set(actual))} unique keys"
        )


    dataset_by_key = {(entry['team'], entry['entry_id']): entry for entry in dataset}
    for result in results:
        key = (result['team'], result['entry_id'])
        expected_entry = dataset_by_key[key]
        for field in ('text', 'triples'):
            if result.get(field) != expected_entry[field]:
                raise ValueError(f"Input mismatch for {key}: {field}")


def existing_results_complete(output_file: Path, dataset: List[Dict]) -> bool:
    if not output_file.exists():
        return False
    try:
        with output_file.open(encoding='utf-8') as handle:
            validate_results(json.load(handle), dataset)
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(
            f"Invalid existing output {output_file}; move or remove it before rerunning: {exc}"
        ) from exc
    return True

def load_team_generations(team_dir: str) -> Dict[str, List[str]]:
    team_dir = Path(team_dir)
    team_data = {}
    txt_files = sorted(team_dir.glob('*.txt'))

    for txt_file in txt_files:
        team_name = txt_file.stem
        with open(txt_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f]
        team_data[team_name] = lines

    return team_data


def load_triples(triples_file: str) -> List[List[str]]:
    with open(triples_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    all_triples = []
    for line in lines:
        triples = parse_triple_line(line)
        all_triples.append(triples)
    return all_triples


def create_evaluation_dataset(
        team_data: Dict[str, List[str]],
        triples_data: List[List[str]]
) -> List[Dict]:
    dataset = []
    expected_entries = len(triples_data)
    if not team_data or not expected_entries:
        raise ValueError("Team generations and triples must be nonempty")

    for team_name, texts in team_data.items():
        if len(texts) != expected_entries:
            raise ValueError(
                f"{team_name}: expected {expected_entries} text rows, got {len(texts)}"
            )
        for i in range(len(texts)):
            if i == MISSING_ENTRY_ID:
                continue
            entry = {
                'id': f'{team_name}_{i}',
                'team': team_name,
                'entry_id': i,
                'text': texts[i],
                'triples': triples_data[i],
            }
            dataset.append(entry)

    return dataset


def load_model(checkpoint_config: Dict):
    """Load model based on checkpoint configuration"""
    lora_checkpoint = checkpoint_config['lora_checkpoint']
    if not Path(lora_checkpoint).is_dir():
        lora_checkpoint = snapshot_download(repo_id=lora_checkpoint)
    torch.set_grad_enabled(False)

    model, tokenizer = get_model_tokenizer(
        checkpoint_config['base_model'],
        model_kwargs={
            'device_map': 'auto',
            'torch_dtype': torch.bfloat16,
        })

    model = Swift.from_pretrained(
        model,
        model_id=lora_checkpoint,
        adapter_name='default')
    model.eval()

    template = get_template(
        checkpoint_config['template_type'],
        tokenizer,
        default_system=checkpoint_config['system_prompt']
    )
    engine = PtEngine.from_model_template(
        model, template, max_batch_size=checkpoint_config['batch_size']
    )
    request_config = RequestConfig(max_tokens=MAX_TOKENS, temperature=TEMPERATURE)

    return engine, request_config


def run_evaluation(
        dataset: List[Dict],
        engine: PtEngine,
        request_config: RequestConfig,
        batch_size: int
) -> List[Dict]:
    results = []

    for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating"):
        batch = dataset[i:i + batch_size]

        try:
            infer_requests = []
            for entry in batch:
                if not entry['text'] and not entry['triples']:
                    infer_requests.append(None)
                elif not entry['text'] or not entry['triples']:
                    infer_requests.append(None)
                else:
                    query = build_evaluation_query(entry['text'], entry['triples'])
                    infer_requests.append(
                        InferRequest(messages=[{'role': 'user', 'content': query}]))

            valid_requests = [r for r in infer_requests if r is not None]
            if valid_requests:
                resp_list = engine.infer(valid_requests, request_config, use_tqdm=False)
            else:
                resp_list = []

            if len(resp_list) != len(valid_requests):
                raise RuntimeError(
                    f"Expected {len(valid_requests)} responses, got {len(resp_list)}"
                )
            resp_idx = 0
            for entry, req in zip(batch, infer_requests):
                if req is None:
                    if not entry['text'] and not entry['triples']:
                        response = "Both empty - aligned"
                        prediction = {'missing': [], 'extra': [], 'incorrect': []}
                        is_aligned = True
                    elif not entry['text']:
                        response = "Empty text - all triples missing"
                        prediction = {
                            'missing': [format_triple_to_prompt(t) for t in entry['triples']],
                            'extra': [],
                            'incorrect': []
                        }
                        is_aligned = False
                    else:
                        response = "Empty triples - text is extra"
                        prediction = {
                            'missing': [],
                            'extra': ['Text: ' + entry['text']],
                            'incorrect': []
                        }
                        is_aligned = False
                    prompt = "Skipped"
                else:
                    resp = resp_list[resp_idx]
                    resp_idx += 1
                    response = resp.choices[0].message.content
                    prediction = parse_model_response(response)
                    prompt = req.messages[0]['content']
                    is_aligned = sum(len(v) for v in prediction.values()) == 0

                total_errors = sum(len(v) for v in prediction.values())

                result = {
                    'id': entry['id'],
                    'team': entry['team'],
                    'entry_id': entry['entry_id'],
                    'text': entry['text'],
                    'triples': entry['triples'],
                    'formatted_triples': [format_triple_to_prompt(t) for t in entry['triples']] if entry[
                        'triples'] else [],
                    'prompt': prompt,
                    'model_response': response,
                    'parsed_errors': prediction,
                    'error_counts': {
                        'missing': len(prediction['missing']),
                        'extra': len(prediction['extra']),
                        'incorrect': len(prediction['incorrect']),
                        'total': total_errors
                    },
                    'is_aligned': is_aligned
                }
                results.append(result)

        except Exception as e:
            print(f"\n❌ Error at batch {i}: {e}")
            import traceback
            traceback.print_exc()
            raise

    validate_results(results, dataset)
    return results


def save_results(results: List[Dict], output_file: str):
    """Save evaluation results to JSON file"""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def main():
    """Main batch evaluation pipeline"""
    args = parse_args()
    print("BATCH EVALUATION - WEBNLG20 CHECKPOINTS")

    set_random_seed(RANDOM_SEED)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_configs = build_dataset_configs(args)

    print(f"\n📊 Configuration:")
    print(f"   Datasets: {len(dataset_configs)}")
    print(f"   Checkpoints: {len(CHECKPOINTS)}")
    print(f"   Total evaluations: {len(dataset_configs) * len(CHECKPOINTS)}")
    print(f"   Random seed: {RANDOM_SEED}")
    print("   CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "not set"))
    print(f"   Team dir: {args.team_dir}")
    print(f"   Triples file: {args.triples_file}")
    print(f"   Output dir: {args.output_dir}")

    # Load all datasets first
    print("LOADING DATASETS")

    datasets_loaded = []
    for dataset_config in dataset_configs:
        print(f"\n📂 Loading {dataset_config['name']}...")
        team_data = load_team_generations(dataset_config['team_dir'])
        triples_data = load_triples(dataset_config['triples_file'])
        dataset = create_evaluation_dataset(team_data, triples_data)

        print(f"   ✅ {len(dataset)} entries loaded")

        datasets_loaded.append({
            'name': dataset_config['name'],
            'data': dataset,
            'output_dir': dataset_config['output_dir']
        })

    # Process each checkpoint
    total_evals = len(CHECKPOINTS) * len(datasets_loaded)
    current_eval = 0

    for checkpoint_idx, checkpoint_config in enumerate(CHECKPOINTS, 1):
        print(f"\n" + "=" * 80)
        print(f"CHECKPOINT {checkpoint_idx}/{len(CHECKPOINTS)}: {checkpoint_config['name']}")
        print("=" * 80)
        print(f"   Model: {checkpoint_config['base_model']}")
        print(f"   LoRA: {checkpoint_config['lora_checkpoint']}")
        print(f"   Batch size: {checkpoint_config['batch_size']}")

        existing_outputs = [
            args.skip_existing and existing_results_complete(
                Path(dataset_info['output_dir']) / f"{checkpoint_config['name']}.json",
                dataset_info['data'],
            )
            for dataset_info in datasets_loaded
        ]
        if all(existing_outputs):
            current_eval += len(datasets_loaded)
            print("   ⏭️  Skipping model - all outputs already exist")
            continue

        # Load model once for all datasets
        print(f"\n🔧 Loading model...")
        try:
            set_random_seed(RANDOM_SEED)
            engine, request_config = load_model(checkpoint_config)
            print(f"   ✅ Model loaded")
        except Exception as e:
            print(f"   ❌ Failed to load model: {e}")
            raise

        # Evaluate on each dataset
        for dataset_info in datasets_loaded:
            current_eval += 1
            print(f"\n{'─' * 80}")
            print(f"[{current_eval}/{total_evals}] Evaluating on {dataset_info['name']}")
            print(f"{'─' * 80}")

            # Output file
            output_file = Path(dataset_info['output_dir']) / f"{checkpoint_config['name']}.json"

            if args.skip_existing and existing_results_complete(output_file, dataset_info['data']):
                print(f"   ⏭️  Skipping - output already exists: {output_file}")
                continue

            try:
                # Run evaluation
                results = run_evaluation(
                    dataset_info['data'],
                    engine,
                    request_config,
                    checkpoint_config['batch_size']
                )

                # Save results
                validate_results(results, dataset_info['data'])
                save_results(results, str(output_file))

                # Quick stats
                aligned = sum(1 for r in results if r['is_aligned'])
                print(f"   ✅ Complete: {aligned}/{len(results)} aligned ({aligned / len(results) * 100:.1f}%)")
                print(f"   💾 Saved to: {output_file}")

            except Exception as e:
                print(f"   ❌ Error: {e}")
                import traceback
                traceback.print_exc()
                raise

        # Clean up GPU memory
        del engine
        torch.cuda.empty_cache()
        print(f"\n🧹 Cleaned up GPU memory")

    print(f"\n" + "=" * 80)
    print("✅ BATCH EVALUATION COMPLETE!")
    print("=" * 80)
    print(f"   Total evaluations: {current_eval}/{total_evals}")


if __name__ == '__main__':
    main()
