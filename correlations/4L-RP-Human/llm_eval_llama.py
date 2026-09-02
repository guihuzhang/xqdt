#!/usr/bin/env python3
"""
LLM Evaluation on William's 50 Annotations
Evaluates all trained LLM checkpoints and computes P/R/F1 with correlation analysis
"""

import json
import os
import re
import random
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict
from scipy.stats import pearsonr, spearmanr, kendalltau
from huggingface_hub import snapshot_download

os.environ.setdefault("USE_HF", "1")

from swift.llm import (
    PtEngine, RequestConfig, get_model_tokenizer, get_template, InferRequest
)
from swift.tuners import Swift

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_JSON = SCRIPT_DIR / "by_language" / "english.json"
OUTPUT_DIR = SCRIPT_DIR / "xqdt_results"
MAX_TOKENS = 1024
TEMPERATURE = 0.3
GPU_ID = '0'
RANDOM_SEED = 2023

# ============================================================================
# ALL CHECKPOINTS - TABLE ENTRIES FIRST, THEN COMMENTED ONES
# ============================================================================

CHECKPOINTS = [
    # ========================================================================
    # CHECKPOINTS USED 
    # ========================================================================
    {
        'base_model': 'meta-llama/Llama-3.2-1B-Instruct',
        'lora_checkpoint': 'Loria-MosAIk/xqdt-webnlg-llama3.2-1b',
        'template_type': 'llama3_2',
        'system_prompt': "Identify extra, missing and incorrect triples precisely.",
        'batch_size': 4,
        'name': 'llama3.2_1b_8864'
    },
    {
        'base_model': 'meta-llama/Llama-3.2-3B-Instruct',
        'lora_checkpoint': 'Loria-MosAIk/xqdt-webnlg-llama3.2-3b',
        'template_type': 'llama3_2',
        'system_prompt': "Identify extra, missing and incorrect triples precisely.",
        'batch_size': 2,
        'name': 'llama3.2_3b_5540'
    },
    {
        'base_model': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
        'lora_checkpoint': 'Loria-MosAIk/xqdt-webnlg-llama3.1-8b',
        'template_type': 'llama3_2',
        'system_prompt': "Identify extra, missing and incorrect triples precisely.",
        'batch_size': 1,
        'name': 'llama3.1_8b_5263'
    },
    # {
    #     'base_model': 'google/gemma-3-270m-it',
    #     'lora_checkpoint': '../../saved_weight/output_gemma3_270correct0warm/v0-20251211-213646/checkpoint-16620',
    #     'template_type': 'gemma3_text',
    #     'system_prompt': None,
    #     'batch_size': 8,
    #     'name': 'gemma3_270m_16620'
    # },
    # {
    #     'base_model': 'google/gemma-3-1b-it',
    #     'lora_checkpoint': '../../saved_weight/output_gemma3_1b_correct/v4-20251211-081939/checkpoint-6648',
    #     'template_type': 'gemma3_text',
    #     'system_prompt': None,
    #     'batch_size': 4,
    #     'name': 'gemma3_1b_6648'
    # },
    # {
    #     'base_model': 'google/gemma-3-4b-it',
    #     'lora_checkpoint': '../../saved_weight/output_gemma3_4b_correct/v0-20251213-035913/checkpoint-6648',
    #     'template_type': 'gemma3_text',
    #     'system_prompt': None,
    #     'batch_size': 2,
    #     'name': 'gemma3_4b_6648'
    # },
    # {
    #     'base_model': 'LLM-Research/gemma-3-12b-it',
    #     'lora_checkpoint': '../../saved_weight/output_gemma3_12b_correct/v0-20251215-055610/checkpoint-4432',
    #     'template_type': 'gemma3_text',
    #     'system_prompt': None,
    #     'batch_size': 2,
    #     'name': 'gemma3_12b_4432'
    # },
    # {
    #     'base_model': 'Qwen/Qwen3-0.6B',
    #     'lora_checkpoint': '../../saved_weight/output_qwen3_0.6correct0warm/v0-20251211-212729/checkpoint-8864',
    #     'template_type': 'qwen3',
    #     'system_prompt': "Identify extra, missing and incorrect triples precisely.",
    #     'batch_size': 8,
    #     'name': 'qwen3_0.6b_8864'
    # },
    # {
    #     'base_model': 'Qwen/Qwen3-1.7B',
    #     'lora_checkpoint': '../../saved_weight/output_qwen3_1.7correct/v3-20251211-081218/checkpoint-6648',
    #     'template_type': 'qwen3',
    #     'system_prompt': "Identify extra, missing and incorrect triples precisely.",
    #     'batch_size': 4,
    #     'name': 'qwen3_1.7b_6648'
    # },
    # {
    #     'base_model': 'Qwen/Qwen3-4B',
    #     'lora_checkpoint': '../../saved_weight/output_qwen3_4b_correct/v3-20251213-032130/checkpoint-4432',
    #     'template_type': 'qwen3',
    #     'system_prompt': "Identify extra, missing and incorrect triples precisely.",
    #     'batch_size': 2,
    #     'name': 'qwen3_4b_4432'
    # },
    # {
    #     'base_model': 'Qwen/Qwen3-8B',
    #     'lora_checkpoint': '../../saved_weight/output_qwen3_8b_correct/v3-20251215-185808/checkpoint-5263',
    #     'template_type': 'qwen3',
    #     'system_prompt': "Identify extra, missing and incorrect triples precisely.",
    #     'batch_size': 2,
    #     'name': 'qwen3_8b_5263'
    # },
    # {
    #     'base_model': 'Qwen/Qwen3-14B',
    #     'lora_checkpoint': '../../saved_weight/output_qwen3_14b_correct/v2-20251214-203429/checkpoint-6648',
    #     'template_type': 'qwen3',
    #     'system_prompt': "Identify extra, missing and incorrect triples precisely.",
    #     'batch_size': 2,
    #     'name': 'qwen3_14b_6648'
    # },
    # ========================================================================
    # COMMENTED CHECKPOINTS (backups)
    # ========================================================================

    # # Gemma3-270M-14404
    # {
    #     'base_model': 'google/gemma-3-270m-it',
    #     'lora_checkpoint': '../../saved_weight/output_gemma3_270correct0warm/v0-20251211-213646/checkpoint-14404',
    #     'template_type': 'gemma3_text',
    #     'system_prompt': None,
    #     'batch_size': 8,
    #     'name': 'gemma3_270m_14404'
    # },

    # # Gemma3-1B-8864
    # {
    #     'base_model': 'google/gemma-3-1b-it',
    #     'lora_checkpoint': '../../saved_weight/output_gemma3_1b_correct/v4-20251211-081939/checkpoint-8864',
    #     'template_type': 'gemma3_text',
    #     'system_prompt': None,
    #     'batch_size': 4,
    #     'name': 'gemma3_1b_8864'
    # },

    # # Gemma3-4B-4432
    # {
    #     'base_model': 'google/gemma-3-4b-it',
    #     'lora_checkpoint': '../../saved_weight/output_gemma3_4b_correct/v0-20251213-035913/checkpoint-4432',
    #     'template_type': 'gemma3_text',
    #     'system_prompt': None,
    #     'batch_size': 2,
    #     'name': 'gemma3_4b_4432'
    # },

    # # Gemma3-12B-6648
    # {
    #     'base_model': 'LLM-Research/gemma-3-12b-it',
    #     'lora_checkpoint': '../../saved_weight/output_gemma3_12b_correct/v0-20251215-055610/checkpoint-6648',
    #     'template_type': 'gemma3_text',
    #     'system_prompt': None,
    #     'batch_size': 2,
    #     'name': 'gemma3_12b_6648'
    # },

    # # Qwen3-0.6B-6648
    # {
    #     'base_model': 'Qwen/Qwen3-0.6B',
    #     'lora_checkpoint': '../../saved_weight/output_qwen3_0.6correct0warm/v0-20251211-212729/checkpoint-6648',
    #     'template_type': 'qwen3',
    #     'system_prompt': "Identify extra, missing and incorrect triples precisely.",
    #     'batch_size': 8,
    #     'name': 'qwen3_0.6b_6648'
    # },

    # # Qwen3-1.7B-8864
    # {
    #     'base_model': 'Qwen/Qwen3-1.7B',
    #     'lora_checkpoint': '../../saved_weight/output_qwen3_1.7correct/v3-20251211-081218/checkpoint-8864',
    #     'template_type': 'qwen3',
    #     'system_prompt': "Identify extra, missing and incorrect triples precisely.",
    #     'batch_size': 4,
    #     'name': 'qwen3_1.7b_8864'
    # },

    # # Qwen3-4B-6648
    # {
    #     'base_model': 'Qwen/Qwen3-4B',
    #     'lora_checkpoint': '../../saved_weight/output_qwen3_4b_correct/v3-20251213-032130/checkpoint-6648',
    #     'template_type': 'qwen3',
    #     'system_prompt': "Identify extra, missing and incorrect triples precisely.",
    #     'batch_size': 2,
    #     'name': 'qwen3_4b_6648'
    # },

    # # Qwen3-8B-7479
    # {
    #     'base_model': 'Qwen/Qwen3-8B',
    #     'lora_checkpoint': '../../saved_weight/output_qwen3_8b_correct/v3-20251215-185808/checkpoint-7479',
    #     'template_type': 'qwen3',
    #     'system_prompt': "Identify extra, missing and incorrect triples precisely.",
    #     'batch_size': 2,
    #     'name': 'qwen3_8b_7479'
    # },

    # # Qwen3-8B-7756 (old training)
    # {
    #     'base_model': 'Qwen/Qwen3-8B',
    #     'lora_checkpoint': '../../saved_weight/output_qwen3_8b_correct/v0-20251213-205905/checkpoint-7756',
    #     'template_type': 'qwen3',
    #     'system_prompt': "Identify extra, missing and incorrect triples precisely.",
    #     'batch_size': 2,
    #     'name': 'qwen3_8b_7756'
    # },

    # # Qwen3-8B-9972 (old training)
    # {
    #     'base_model': 'Qwen/Qwen3-8B',
    #     'lora_checkpoint': '../../saved_weight/output_qwen3_8b_correct/v0-20251213-205905/checkpoint-9972',
    #     'template_type': 'qwen3',
    #     'system_prompt': "Identify extra, missing and incorrect triples precisely.",
    #     'batch_size': 2,
    #     'name': 'qwen3_8b_9972'
    # },

    # # Qwen3-14B-4432
    # {
    #     'base_model': 'Qwen/Qwen3-14B',
    #     'lora_checkpoint': '../../saved_weight/output_qwen3_14b_correct/v2-20251214-203429/checkpoint-4432',
    #     'template_type': 'qwen3',
    #     'system_prompt': "Identify extra, missing and incorrect triples precisely.",
    #     'batch_size': 2,
    #     'name': 'qwen3_14b_4432'
    # },

    # # Llama3.2-1B-6648
    # {
    #     'base_model': 'LLM-Research/Llama-3.2-1B-Instruct',
    #     'lora_checkpoint': '../../saved_weight/output_llama3_1b_correct/v2-20251213-021354/checkpoint-6648',
    #     'template_type': 'llama3_2',
    #     'system_prompt': "Identify extra, missing and incorrect triples precisely.",
    #     'batch_size': 4,
    #     'name': 'llama3.2_1b_6648'
    # },

    # # Llama3.2-3B-5540
    # {
    #     'base_model': 'LLM-Research/Llama-3.2-3B-Instruct',
    #     'lora_checkpoint': '../../saved_weight/output_llama3_3b_correct/v1-20251213-203738/checkpoint-5540',
    #     'template_type': 'llama3_2',
    #     'system_prompt': "Identify extra, missing and incorrect triples precisely.",
    #     'batch_size': 4,
    #     'name': 'llama3.2_3b_5540'
    # },

    # # Llama3.1-8B-4432 (old training)
    # {
    #     'base_model': 'LLM-Research/Meta-Llama-3.1-8B-Instruct',
    #     'lora_checkpoint': '../../saved_weight/output_llama3.1_8b_correct/v1-20251214-200900/checkpoint-4432',
    #     'template_type': 'llama3_2',
    #     'system_prompt': "Identify extra, missing and incorrect triples precisely.",
    #     'batch_size': 2,
    #     'name': 'llama3.1_8b_4432'
    # },

    # # Llama3.1-8B-5263 (old training)
    # {
    #     'base_model': 'LLM-Research/Meta-Llama-3.1-8B-Instruct',
    #     'lora_checkpoint': '../../saved_weight/output_llama3.1_8b_correct/v1-20251214-200900/checkpoint-5263',
    #     'template_type': 'llama3_2',
    #     'system_prompt': "Identify extra, missing and incorrect triples precisely.",
    #     'batch_size': 2,
    #     'name': 'llama3.1_8b_5263'
    # },

    # # Llama3.1-8B-7756 (new training)
    # {
    #     'base_model': 'LLM-Research/Meta-Llama-3.1-8B-Instruct',
    #     'lora_checkpoint': '../../saved_weight/output_llama3.1_8b_correct/v3-20251217-054418/checkpoint-7756',
    #     'template_type': 'llama3_2',
    #     'system_prompt': "Identify extra, missing and incorrect triples precisely.",
    #     'batch_size': 2,
    #     'name': 'llama3.1_8b_7756'
    # },
]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def set_random_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def clean_entity(entity: str) -> str:
    """Clean entity string"""
    entity = re.sub(r'@[a-z]{2,3}$', '', entity)
    entity = re.sub(r'\^\^xsd:\w+', '', entity)
    entity = re.sub(r'\^\^<http://[^>]+>', '', entity)
    entity = entity.strip('"\'')
    entity = entity.replace('_', ' ')
    return entity.strip()


def clean_predicate(predicate: str) -> str:
    """Clean predicate string"""
    predicate = predicate.replace('/', ' ')
    predicate = re.sub(r'([a-z])([A-Z])', r'\1 \2', predicate)
    predicate = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', predicate)
    return predicate.strip().lower()


def format_triple_to_prompt(triple_str: str) -> str:
    """Format triple for prompt"""
    parts = [p.strip() for p in triple_str.split('|')]
    if len(parts) != 3:
        return triple_str
    subject, predicate, obj = parts
    subject = clean_entity(subject)
    predicate = clean_predicate(predicate)
    obj = clean_entity(obj)
    return f"[S] {subject} [P] {predicate} [O] {obj}"


def build_evaluation_query(text: str, triples: List[str]) -> str:
    """Build evaluation query for LLM"""
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
    """Parse model response to extract errors"""
    result = {'missing': [], 'extra': [], 'incorrect': []}
    if not response:
        return result

    if any(phrase in response.lower() for phrase in ['all correct', 'no error', 'aligned']):
        return result

    lines = response.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line or '|' not in line:
            continue
        if any(keyword in line.lower() for keyword in ['type', '----', '====', '---']):
            continue

        parts = [p.strip() for p in line.split('|')]
        parts = [p for p in parts if p]
        if len(parts) < 2:
            continue

        error_type = parts[0].strip().lower()
        triple = ' '.join(parts[1:]).strip()
        if not triple:
            continue

        if 'missing' in error_type or 'omission' in error_type:
            result['missing'].append(triple)
        elif 'extra' in error_type or 'additional' in error_type:
            result['extra'].append(triple)
        elif 'incorrect' in error_type or 'wrong' in error_type or 'error' in error_type:
            result['incorrect'].append(triple)

    return result


def compute_prf_from_errors(parsed_errors: Dict[str, List[str]], num_triples: int) -> Dict[str, float]:
    """
    Compute final P/R/F1 from parsed errors.

    Incorrect triples count against both precision and recall:
    TP = |D| - |missing| - |incorrect|
    FP = |extra| + |incorrect|
    FN = |missing| + |incorrect|
    """
    num_missing = len(parsed_errors['missing'])
    num_extra = len(parsed_errors['extra'])
    num_incorrect = len(parsed_errors['incorrect'])

    TP = max(0, num_triples - num_missing - num_incorrect)
    FP = num_extra + num_incorrect
    FN = num_missing + num_incorrect

    if TP + FP > 0:
        precision = TP / (TP + FP)
    else:
        precision = 0.0

    if TP + FN > 0:
        recall = TP / (TP + FN)
    else:
        recall = 0.0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'num_missing': num_missing,
        'num_extra': num_extra,
        'num_incorrect': num_incorrect,
    }


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model(checkpoint_config: Dict):
    """Load LLM model"""
    torch.set_grad_enabled(False)
    adapter_path = snapshot_download(repo_id=checkpoint_config['lora_checkpoint'])

    model, tokenizer = get_model_tokenizer(
        checkpoint_config['base_model'],
        model_kwargs={
            'device_map': 'auto',
            'torch_dtype': torch.bfloat16,
        })

    model = Swift.from_pretrained(
        model,
        model_id=adapter_path,
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


# ============================================================================
# EVALUATION
# ============================================================================

def run_evaluation(data: List[Dict], engine: PtEngine, request_config: RequestConfig, batch_size: int) -> List[Dict]:
    """Run evaluation on dataset"""
    results = []

    for i in tqdm(range(0, len(data), batch_size), desc="Evaluating"):
        batch = data[i:i + batch_size]

        try:
            infer_requests = []
            for entry in batch:
                triples = [t.strip() for t in entry['graph'].split('<br>') if t.strip()]
                query = build_evaluation_query(entry['text'], triples)
                infer_requests.append(
                    InferRequest(messages=[{'role': 'user', 'content': query}]))

            resp_list = engine.infer(infer_requests, request_config, use_tqdm=False)

            for entry, resp in zip(batch, resp_list):
                response = resp.choices[0].message.content
                parsed_errors = parse_model_response(response)

                triples = [t.strip() for t in entry['graph'].split('<br>') if t.strip()]
                num_triples = len(triples)

                prf = compute_prf_from_errors(parsed_errors, num_triples)

                result = {
                    'id': entry['id'],
                    'question_id': entry['question_id'],
                    'text': entry['text'],
                    'graph': entry['graph'],
                    'num_triples': num_triples,
                    'model_response': response,
                    'parsed_errors': parsed_errors,
                    'error_counts': {
                        'missing': len(parsed_errors['missing']),
                        'extra': len(parsed_errors['extra']),
                        'incorrect': len(parsed_errors['incorrect']),
                    },
                    'precision': prf['precision'],
                    'recall': prf['recall'],
                    'f1': prf['f1'],
                    'TP': prf['TP'],
                    'FP': prf['FP'],
                    'FN': prf['FN'],
                    'prf_stats': {
                        'num_missing': prf['num_missing'],
                        'num_extra': prf['num_extra'],
                        'num_incorrect': prf['num_incorrect'],
                    },
                    'human_precision_list': entry.get('precision_list', []),
                    'human_recall_list': entry.get('recall_list', []),
                }
                results.append(result)

        except Exception as e:
            print(f"\n❌ Error at batch {i}: {e}")
            import traceback
            traceback.print_exc()
            continue

    return results


# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================

def compute_correlations(results: List[Dict], model_name: str, n_bootstrap: int = 1000):
    """Compute correlations with bootstrap"""
    print(f"\n{'=' * 70}")
    print(f"Computing Correlations: {model_name}")
    print('=' * 70)

    model_precision = []
    model_recall = []
    model_f1 = []
    human_precision_avg = []
    human_recall_avg = []
    human_f1_avg = []

    for item in results:
        if len(item['human_precision_list']) == 0:
            continue

        model_precision.append(item['precision'])
        model_recall.append(item['recall'])
        model_f1.append(item['f1'])

        h_prec = np.mean(item['human_precision_list'])
        h_rec = np.mean(item['human_recall_list'])

        if h_prec + h_rec > 0:
            h_f1 = (2 * h_prec * h_rec) / (h_prec + h_rec)
        else:
            h_f1 = 0.0

        human_precision_avg.append(h_prec)
        human_recall_avg.append(h_rec)
        human_f1_avg.append(h_f1)

    if len(model_precision) == 0:
        print("⚠️  No valid samples!")
        return None

    model_precision = np.array(model_precision)
    model_recall = np.array(model_recall)
    model_f1 = np.array(model_f1)
    human_precision_avg = np.array(human_precision_avg)
    human_recall_avg = np.array(human_recall_avg)
    human_f1_avg = np.array(human_f1_avg)

    # Original correlations
    pearson_prec = pearsonr(model_precision, human_precision_avg)
    pearson_rec = pearsonr(model_recall, human_recall_avg)
    pearson_f1 = pearsonr(model_f1, human_f1_avg)

    spearman_prec = spearmanr(model_precision, human_precision_avg)
    spearman_rec = spearmanr(model_recall, human_recall_avg)
    spearman_f1 = spearmanr(model_f1, human_f1_avg)

    kendall_prec = kendalltau(model_precision, human_precision_avg)
    kendall_rec = kendalltau(model_recall, human_recall_avg)
    kendall_f1 = kendalltau(model_f1, human_f1_avg)

    rmse_prec = np.sqrt(np.mean((model_precision - human_precision_avg) ** 2))
    rmse_rec = np.sqrt(np.mean((model_recall - human_recall_avg) ** 2))
    rmse_f1 = np.sqrt(np.mean((model_f1 - human_f1_avg) ** 2))

    print("📊 Correlations:")
    print(f"   Pearson:  P={pearson_prec[0]:.4f}, R={pearson_rec[0]:.4f}, F1={pearson_f1[0]:.4f}")
    print(f"   Spearman: P={spearman_prec[0]:.4f}, R={spearman_rec[0]:.4f}, F1={spearman_f1[0]:.4f}")
    print(f"   Kendall:  P={kendall_prec[0]:.4f}, R={kendall_rec[0]:.4f}, F1={kendall_f1[0]:.4f}")
    print(f"   RMSE:     P={rmse_prec:.4f}, R={rmse_rec:.4f}, F1={rmse_f1:.4f}")

    return {
        'model': model_name,
        'n_samples': len(model_precision),
        'pearson': {'precision': pearson_prec[0], 'recall': pearson_rec[0], 'f1': pearson_f1[0]},
        'spearman': {'precision': spearman_prec[0], 'recall': spearman_rec[0], 'f1': spearman_f1[0]},
        'kendall': {'precision': kendall_prec[0], 'recall': kendall_rec[0], 'f1': kendall_f1[0]},
        'rmse': {'precision': rmse_prec, 'recall': rmse_rec, 'f1': rmse_f1}
    }


# ============================================================================
# MAIN
# ============================================================================

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main evaluation pipeline"""
    print("=" * 70)
    print("LLM EVALUATION - ALL CHECKPOINTS")
    print("=" * 70)

    set_random_seed(RANDOM_SEED)
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"\n📂 Loading: {INPUT_JSON}")
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"📊 Total samples: {len(data)}")

    # Process each checkpoint
    all_correlations = []

    for idx, checkpoint_config in enumerate(CHECKPOINTS, 1):
        print(f"\n{'=' * 70}")
        print(f"[{idx}/{len(CHECKPOINTS)}] {checkpoint_config['name']}")
        print('=' * 70)

        # Output files
        results_file = OUTPUT_DIR / f"{checkpoint_config['name']}_results.json"
        correlation_file = OUTPUT_DIR / f"{checkpoint_config['name']}_correlation.json"

        # Load model
        print("🔧 Loading model...")
        try:
            engine, request_config = load_model(checkpoint_config)
            print("✅ Model loaded!")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            continue

        # Run evaluation
        print("🚀 Running evaluation...")
        try:
            results = run_evaluation(data, engine, request_config, checkpoint_config['batch_size'])

            # Save results
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"💾 Results saved: {results_file}")

            # Compute correlations
            correlation = compute_correlations(results, checkpoint_config['name'])
            if correlation:
                all_correlations.append(correlation)

                with open(correlation_file, 'w', encoding='utf-8') as f:
                    json.dump(correlation, f, indent=2, ensure_ascii=False)
                print(f"💾 Correlation saved: {correlation_file}")

        except Exception as e:
            print(f"❌ Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Clean up
        del engine
        torch.cuda.empty_cache()

    # Save summary
    summary_file = OUTPUT_DIR / "summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_correlations, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Summary saved: {summary_file}")

    print(f"\n{'=' * 70}")
    print("✅ EVALUATION COMPLETE!")
    print('=' * 70)


if __name__ == '__main__':
    main()
