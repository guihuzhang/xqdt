#!/usr/bin/env python3
"""
LLM Evaluation on E2E Human Ratings (converted.json)
Runs trained checkpoints on all (MR, sys) pairs; saves results first,
then computes sample-level and system-level correlations against human fine_score.
"""

import gc
import json
import os
import random
import re
import traceback
import warnings
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
from huggingface_hub import snapshot_download
from scipy.stats import pearsonr, spearmanr, kendalltau
from swift.llm import (
    PtEngine, RequestConfig, get_model_tokenizer, get_template, InferRequest
)
from swift.tuners import Swift
from tqdm import tqdm


# Configuration

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_JSON = (SCRIPT_DIR / "human_ratings/converted.json").resolve()
OUTPUT_DIR = (SCRIPT_DIR / "xqdt_results").resolve()
MAX_TOKENS = 1024
TEMPERATURE = 0.3
GPU_ID = os.environ.get('CUDA_VISIBLE_DEVICES', '')
RANDOM_SEED = 2023

# ALL CHECKPOINTS
CHECKPOINTS = [
    {
        'base_model': 'LLM-Research/Meta-Llama-3.1-8B-Instruct',
        'lora_checkpoint': 'Loria-MosAIk/xqdt-e2e-llama3.1-8b',
        'template_type': 'llama3_2',
        'system_prompt': "Identify extra, missing and incorrect triples precisely.",
        'fixed_batch_size': 1,
        'name': 'llama3.1_8b_8680'
    },
    {
        'base_model': 'LLM-Research/Llama-3.2-3B-Instruct',
        'lora_checkpoint': 'Loria-MosAIk/xqdt-e2e-llama3.2-3b',
        'template_type': 'llama3_2',
        'system_prompt': "Identify extra, missing and incorrect triples precisely.",
        'fixed_batch_size': 2,
        'name': 'llama3.2_3b_6975'
    },
    {
        'base_model': 'LLM-Research/Llama-3.2-1B-Instruct',
        'lora_checkpoint': 'Loria-MosAIk/xqdt-e2e-llama3.2-1b',
        'template_type': 'llama3_2',
        'system_prompt': "Identify extra, missing and incorrect triples precisely.",
        'fixed_batch_size': 4,
        'name': 'llama3.2_1b_9765'
    },
]


# utilities

def set_random_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def normalize_slot_name(slot_name: str) -> str:
    """Normalize E2E slot name to lowercase words."""
    slot_name = slot_name.strip().replace('_', ' ')
    slot_name = re.sub(r'([a-z])([A-Z])', r'\1 \2', slot_name)
    slot_name = re.sub(r'([A-Z])([A-Z][a-z])', r'\1 \2', slot_name)
    slot_name = re.sub(r'\s+', ' ', slot_name)
    return slot_name.strip().lower()


def parse_e2e_mr(mr_text: str) -> List[str]:
    """Convert E2E MR slot[value] format to [S] ... [P] ... [O] triples (same as training data).
    Subject = value of 'name' slot; predicate = camelCase slot name -> lowercase with spaces.
    'name[Blue Spice], eatType[coffee shop], area[city centre]'
    -> ['[S] Blue Spice [P] eat type [O] coffee shop', '[S] Blue Spice [P] area [O] city centre']
    The 'name' slot defines the subject and is not emitted as a separate triple.
    """
    slot_pairs = [
        (m.group(1).strip(), m.group(2).strip())
        for m in re.finditer(r'\s*([^,\[]+?)\s*\[([^\]]*)\]\s*(?:,|$)', mr_text)
    ]
    subject = next(
        (value for slot, value in slot_pairs if normalize_slot_name(slot) == 'name' and value),
        'restaurant')

    triples = []
    for raw_slot, value in slot_pairs:
        if not value:
            continue
        slot = normalize_slot_name(raw_slot)
        if slot == 'name':
            continue
        triples.append(f'[S] {subject} [P] {slot} [O] {value}')
    return triples


def build_evaluation_query(text: str, triples: List[str]) -> str:
    """Build evaluation query using SPO triple format, same as training data."""
    triples_formatted = '\n'.join(
        f'{i}. {t}'
        for i, t in enumerate(triples, start=1))
    return (
        f"Verify if the triples align with the text. "
        f"Find missing, extra, or incorrect triples.\n"
        f"TEXT: {text}\n"
        f"TRIPLES:\n{triples_formatted}\n"
        f"Output as markdown table with Type and Triple columns."
    )


def normalize_for_keyword_match(text: str) -> str:
    """Lowercase and normalize punctuation/whitespace for robust phrase matching."""
    normalized = text.lower()
    normalized = re.sub(r'[_`~*#>\[\](){}|:;,.!?/\\-]+', ' ', normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return normalized


def parse_model_response(response: str) -> Dict[str, List[str]]:
    """Parse model response markdown table to extract error types"""
    result = {'missing': [], 'extra': [], 'incorrect': []}
    if not response or not response.strip():
        raise ValueError("Empty model response")

    normalized_response = normalize_for_keyword_match(response)
    negative_all_correct_patterns = (
        r'\bnot\s+all\s+correct\b',
        r'\bnot\s+correct\b',
        r'\bnot\s+aligned\b',
        r'\bnot\s+fully\s+aligned\b',
        r'\bnot\s+completely\s+aligned\b',
        r'\bnot\s+perfectly\s+aligned\b',
        r'\bmisaligned\b',
        r'\bhas\s+errors?\b',
        r'\bwith\s+errors?\b',
        r'\bcontains?\s+errors?\b',
    )
    positive_all_correct_patterns = (
        r'\ball\s+correct\b',
        r'\bno\s+errors?\b',
        r'\bno\s+alignment\s+errors?\b',
        r'\bno\s+issues?\b',
        r'\bfully\s+aligned\b',
        r'\bcompletely\s+aligned\b',
        r'\bperfectly\s+aligned\b',
    )

    has_negative_marker = any(
        re.search(pattern, normalized_response) for pattern in negative_all_correct_patterns
    )
    has_positive_marker = any(
        re.search(pattern, normalized_response) for pattern in positive_all_correct_patterns
    )
    for line in response.strip().split('\n'):
        line = line.strip()
        if not line or '|' not in line:
            continue
        parts = [p.strip() for p in line.split('|') if p.strip()]
        if len(parts) < 2:
            continue
        lowered = [p.lower() for p in parts]
        if lowered[0] == 'type' and lowered[1].startswith('triple'):
            continue
        if all(re.fullmatch(r':?-{3,}:?', p) for p in lowered):
            continue
        error_type = parts[0].lower()
        attr = ' '.join(parts[1:]).strip()
        if not attr:
            continue
        if 'missing' in error_type or 'omission' in error_type:
            result['missing'].append(attr)
        elif 'extra' in error_type or 'additional' in error_type:
            result['extra'].append(attr)
        elif 'incorrect' in error_type or 'wrong' in error_type or 'error' in error_type:
            result['incorrect'].append(attr)

    if not any(result.values()) and has_positive_marker and not has_negative_marker:
        return result

    if not any(result.values()):
        raise ValueError("Unrecognized model response; no valid error rows or all-correct marker")

    for key in result:
        result[key] = list(dict.fromkeys(result[key]))

    return result


def compute_prf_from_errors(parsed_errors: Dict[str, List[str]], num_slots: int) -> Dict[str, float]:
    """Compute P/R/F1 from parsed errors.
    TP = correctly covered slots; FP = extra + incorrect; FN = missing + incorrect
    """
    num_missing = len(parsed_errors['missing'])
    num_extra = len(parsed_errors['extra'])
    num_incorrect = len(parsed_errors['incorrect'])

    TP = max(0, num_slots - num_missing - num_incorrect)
    FP = num_extra + num_incorrect
    FN = num_missing + num_incorrect

    precision = TP / (TP + FP) if TP + FP > 0 else 0.0
    recall    = TP / (TP + FN) if TP + FN > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return {'precision': precision, 'recall': recall, 'f1': f1, 'TP': TP, 'FP': FP, 'FN': FN}


def flatten_data(converted: Dict) -> List[Dict]:
    """Flatten converted.json into a flat list of (mr_id, sys_name) samples.
    Each sample includes human quality and naturalness fine_score lists.
    """
    samples = []
    for mr_id, mr_data in converted.items():
        mr_text = mr_data['mr']
        for sys_name, sys_data in mr_data['systems'].items():
            quality_annotations = sys_data.get('quality', [])
            quality_scores = [a['fine_score'] for a in quality_annotations]
            quality_coarse_labels = [
                re.sub(r'\s+', ' ', str(a.get('coarse_label', '')).strip().lower())
                for a in quality_annotations
                if str(a.get('coarse_label', '')).strip()
            ]
            quality_is_ok = [label == 'ok' for label in quality_coarse_labels]
            quality_ok_ratio = (
                float(np.mean(quality_is_ok)) if quality_is_ok else None
            )
            quality_coarse_counts = dict(Counter(quality_coarse_labels))
            quality_majority_coarse = (
                sorted(
                    quality_coarse_counts.items(),
                    key=lambda item: (-item[1], item[0])
                )[0][0]
                if quality_coarse_labels else None
            )

            naturalness_scores = [a['fine_score'] for a in sys_data.get('naturalness', [])]
            samples.append({
                'id': f'{mr_id}_{sys_name}',
                'mr_id': mr_id,
                'sys_name': sys_name,
                'mr': mr_text,
                'output': sys_data['output'],
                'human_quality_scores': quality_scores,       # raw 0-100 scores
                'human_quality_coarse_labels': quality_coarse_labels,
                'human_quality_coarse_counts': quality_coarse_counts,
                'human_quality_is_ok': quality_is_ok,
                'human_quality_ok_ratio': quality_ok_ratio,
                'human_quality_majority_coarse': quality_majority_coarse,
                'human_quality_majority_is_ok': (
                    quality_majority_coarse == 'ok'
                    if quality_majority_coarse is not None else None
                ),
                'human_naturalness_scores': naturalness_scores,
            })
    return samples


# MODEL LOADING

def _clear_cuda_memory():
    """Release Python and CUDA memory between checkpoints."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def infer_visible_gpu_count(gpu_id: str) -> int:
    """Infer visible GPU count from config string, fallback to runtime query."""
    ids = [gpu.strip() for gpu in str(gpu_id).split(',') if gpu.strip()]
    if ids:
        return len(ids)
    if torch.cuda.is_available():
        return max(1, torch.cuda.device_count())
    return 1


def resolve_batch_size(checkpoint_config: Dict, visible_gpu_count: int) -> int:
    """Resolve effective global batch size.
    - All Llama checkpoints use fixed_batch_size (8B=1, 3B=2, 1B=4).
    - per_device_batch_size is also supported as a fallback (scaled by gpu count).
    """
    if checkpoint_config.get('fixed_batch_size') is not None:
        return max(1, int(checkpoint_config['fixed_batch_size']))
    if checkpoint_config.get('per_device_batch_size') is not None:
        per_device = max(1, int(checkpoint_config['per_device_batch_size']))
        return per_device * max(1, int(visible_gpu_count))
    raise ValueError(
        f"Checkpoint '{checkpoint_config.get('name')}' must set either "
        f"'fixed_batch_size' or 'per_device_batch_size'."
    )


def _load_base_model(base_model: str):
    """Load base model/tokenizer; retry once with CPU offload on OOM."""
    model_kwargs = {'device_map': 'auto', 'torch_dtype': torch.bfloat16}
    try:
        return get_model_tokenizer(base_model, model_kwargs=model_kwargs)
    except torch.OutOfMemoryError:
        print("\nOOM during model loading. Retrying with CPU offload...")
        _clear_cuda_memory()
        offload_dir = os.environ.get('SWIFT_OFFLOAD_DIR', '/tmp/swift_offload')
        os.makedirs(offload_dir, exist_ok=True)
        retry_kwargs = {
            'device_map': 'auto',
            'torch_dtype': torch.bfloat16,
            'offload_folder': offload_dir,
            'offload_state_dict': True,
        }
        if torch.cuda.is_available():
            max_memory = {}
            for gpu_idx in range(torch.cuda.device_count()):
                total_gib = torch.cuda.get_device_properties(gpu_idx).total_memory / (1024 ** 3)
                max_memory[gpu_idx] = f"{max(1, int(total_gib * 0.85))}GiB"
            max_memory['cpu'] = os.environ.get('SWIFT_MAX_CPU_MEMORY', '120GiB')
            retry_kwargs['max_memory'] = max_memory
        return get_model_tokenizer(base_model, model_kwargs=retry_kwargs)


def load_model(checkpoint_config: Dict, batch_size: int):
    """Load LLM model with LoRA checkpoint"""
    torch.set_grad_enabled(False)

    model, tokenizer = _load_base_model(checkpoint_config['base_model'])

    adapter_path = snapshot_download(repo_id=checkpoint_config['lora_checkpoint'])

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
        model, template, max_batch_size=batch_size
    )
    request_config = RequestConfig(max_tokens=MAX_TOKENS, temperature=TEMPERATURE)

    return engine, request_config


# EVALUATION

def run_evaluation(data: List[Dict], engine, request_config, batch_size: int) -> List[Dict]:
    """Run model inference on all (MR, sys) pairs; embed human scores in results"""
    results = []

    for i in tqdm(range(0, len(data), batch_size), desc="Evaluating"):
        batch = data[i:i + batch_size]

        try:
            infer_requests = []
            batch_triples = []
            for entry in batch:
                triples = parse_e2e_mr(entry['mr'])
                batch_triples.append(triples)
                query = build_evaluation_query(entry['output'], triples)
                infer_requests.append(
                    InferRequest(messages=[{'role': 'user', 'content': query}]))

            resp_list = engine.infer(infer_requests, request_config, use_tqdm=False)

            if len(resp_list) != len(batch):
                raise RuntimeError(f"Expected {len(batch)} responses, received {len(resp_list)}")

            for entry, resp, triples in zip(batch, resp_list, batch_triples):
                response = resp.choices[0].message.content
                parsed_errors = parse_model_response(response)
                num_triples = len(triples)
                prf = compute_prf_from_errors(parsed_errors, num_triples)

                results.append({
                    'id': entry['id'],
                    'mr_id': entry['mr_id'],
                    'sys_name': entry['sys_name'],
                    'mr': entry['mr'],
                    'output': entry['output'],
                    'num_triples': num_triples,
                    'model_response': response,
                    'parsed_errors': parsed_errors,
                    'error_counts': {
                        'missing':   len(parsed_errors['missing']),
                        'extra':     len(parsed_errors['extra']),
                        'incorrect': len(parsed_errors['incorrect']),
                    },
                    'precision': prf['precision'],
                    'recall':    prf['recall'],
                    'f1':        prf['f1'],
                    'TP': prf['TP'],
                    'FP': prf['FP'],
                    'FN': prf['FN'],
                    # human scores stored here so correlation can be computed without re-reading converted.json
                    'human_quality_scores':     entry['human_quality_scores'],
                    'human_quality_coarse_labels': entry['human_quality_coarse_labels'],
                    'human_quality_coarse_counts': entry['human_quality_coarse_counts'],
                    'human_quality_is_ok':         entry['human_quality_is_ok'],
                    'human_quality_ok_ratio':      entry['human_quality_ok_ratio'],
                    'human_quality_majority_coarse': entry['human_quality_majority_coarse'],
                    'human_quality_majority_is_ok':  entry['human_quality_majority_is_ok'],
                    'human_naturalness_scores': entry['human_naturalness_scores'],
                })

        except Exception as e:
            print(f"\n❌ Error at batch {i}: {e}")
            traceback.print_exc()
            raise

    expected = [(str(row['mr_id']), row['sys_name']) for row in data]
    actual = [(str(row['mr_id']), row['sys_name']) for row in results]
    if len(set(expected)) != len(expected) or len(actual) != len(expected) or len(set(actual)) != len(actual) or set(actual) != set(expected):
        raise RuntimeError("Incomplete or duplicate inference results")
    return results


# CORRELATION ANALYSIS

def _bootstrap_ci(values: list, ci: float = 0.95) -> Dict:
    """Compute bootstrap mean and confidence interval from a list of per-iteration values.
    No p-value filtering — all iterations kept (avoids the upward bias in the old WebNLG code).
    """
    if not values:
        return {'mean': 0.0, 'ci_low': 0.0, 'ci_high': 0.0, 'n_boot': 0}
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {'mean': 0.0, 'ci_low': 0.0, 'ci_high': 0.0, 'n_boot': 0}
    lo = (1 - ci) / 2
    hi = 1 - lo
    return {
        'mean':    float(np.mean(arr)),
        'ci_low':  float(np.percentile(arr, lo * 100)),
        'ci_high': float(np.percentile(arr, hi * 100)),
        'n_boot':  int(arr.size),
    }


def _safe_corr(corr_fn, x: np.ndarray, y: np.ndarray) -> float:
    """Return a finite correlation value; fallback to 0.0 when undefined."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            value = corr_fn(x, y)[0]
    except Exception:
        return 0.0
    if value is None:
        return 0.0
    value = float(value)
    if not np.isfinite(value):
        return 0.0
    return value


def _safe_mean(values: list) -> float:
    """Mean over finite values; fallback to 0.0 when list is empty/invalid."""
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr))


def compute_correlations(results: List[Dict], model_name: str, n_bootstrap: int = 1000) -> Dict:
    """Compute sample-level, text-level, and system-level correlations vs human quality fine_score.

    Sample-level:
        Each (MR, sys) pair is one data point. Correlation across all ~6012 pairs.
        Bootstrap resamples pairs.

    Text-level (matching paper methodology):
        For each MR, compute correlation across available systems (model vs human score),
        then average across MRs. Bootstrap resamples MRs (no p-value filtering).

    System-level:
        Average model/human scores per system, correlate across 21 systems.

    Human scores normalized from [0, 100] to [0, 1].
    """
    print(f"\nCorrelations: {model_name}")

    # ---- Build per-MR data structure ----
    # mr_data[mr_id] = {'prec': [...], 'rec': [...], 'f1': [...], 'qual': [...]}
    # one entry per system that has human quality annotations
    mr_data: Dict[str, Dict[str, list]] = defaultdict(
        lambda: {'prec': [], 'rec': [], 'f1': [], 'qual': []})
    sys_model = defaultdict(lambda: {'prec': [], 'rec': [], 'f1': []})
    sys_human: Dict[str, list] = defaultdict(list)
    _prec, _rec, _f1, _qual = [], [], [], []

    for r in results:
        if not r['human_quality_scores']:
            continue
        mid = r['mr_id']
        sys = r['sys_name']
        qual = np.mean(r['human_quality_scores']) / 100.0

        mr_data[mid]['prec'].append(r['precision'])
        mr_data[mid]['rec'].append(r['recall'])
        mr_data[mid]['f1'].append(r['f1'])
        mr_data[mid]['qual'].append(qual)

        sys_model[sys]['prec'].append(r['precision'])
        sys_model[sys]['rec'].append(r['recall'])
        sys_model[sys]['f1'].append(r['f1'])
        sys_human[sys].append(qual)

        _prec.append(r['precision'])
        _rec.append(r['recall'])
        _f1.append(r['f1'])
        _qual.append(qual)

    # ---- Sample-level: all (MR, sys) pairs as individual data points ----
    all_prec = np.array(_prec)
    all_rec  = np.array(_rec)
    all_f1   = np.array(_f1)
    all_qual = np.array(_qual)

    sample_corr = {}
    if len(all_prec) > 1:
        # Point estimates
        sample_corr = {
            'n': int(len(all_prec)),
            'point': {
                'pearson':  {'precision': _safe_corr(pearsonr, all_prec, all_qual),
                             'recall':    _safe_corr(pearsonr, all_rec,  all_qual),
                             'f1':        _safe_corr(pearsonr, all_f1,   all_qual)},
                'spearman': {'precision': _safe_corr(spearmanr, all_prec, all_qual),
                             'recall':    _safe_corr(spearmanr, all_rec,  all_qual),
                             'f1':        _safe_corr(spearmanr, all_f1,   all_qual)},
                'kendall':  {'precision': _safe_corr(kendalltau, all_prec, all_qual),
                             'recall':    _safe_corr(kendalltau, all_rec,  all_qual),
                             'f1':        _safe_corr(kendalltau, all_f1,   all_qual)},
            }
        }
        # Bootstrap over pairs
        n = len(all_prec)
        boot_pear_p,  boot_pear_r,  boot_pear_f  = [], [], []
        boot_spear_p, boot_spear_r, boot_spear_f = [], [], []
        boot_kend_p,  boot_kend_r,  boot_kend_f  = [], [], []
        for _ in range(n_bootstrap):
            idx = np.random.randint(0, n, size=n)
            boot_pear_p.append(_safe_corr(pearsonr, all_prec[idx],   all_qual[idx]))
            boot_pear_r.append(_safe_corr(pearsonr, all_rec[idx],    all_qual[idx]))
            boot_pear_f.append(_safe_corr(pearsonr, all_f1[idx],     all_qual[idx]))
            boot_spear_p.append(_safe_corr(spearmanr, all_prec[idx], all_qual[idx]))
            boot_spear_r.append(_safe_corr(spearmanr, all_rec[idx],  all_qual[idx]))
            boot_spear_f.append(_safe_corr(spearmanr, all_f1[idx],   all_qual[idx]))
            boot_kend_p.append(_safe_corr(kendalltau, all_prec[idx], all_qual[idx]))
            boot_kend_r.append(_safe_corr(kendalltau, all_rec[idx],  all_qual[idx]))
            boot_kend_f.append(_safe_corr(kendalltau, all_f1[idx],   all_qual[idx]))
        sample_corr['bootstrap'] = {
            'pearson':  {'precision': _bootstrap_ci(boot_pear_p),  'recall': _bootstrap_ci(boot_pear_r),  'f1': _bootstrap_ci(boot_pear_f)},
            'spearman': {'precision': _bootstrap_ci(boot_spear_p), 'recall': _bootstrap_ci(boot_spear_r), 'f1': _bootstrap_ci(boot_spear_f)},
            'kendall':  {'precision': _bootstrap_ci(boot_kend_p),  'recall': _bootstrap_ci(boot_kend_r),  'f1': _bootstrap_ci(boot_kend_f)},
        }
        print(f"📊 Sample-level ({sample_corr['n']} pairs):")
        for method in ('pearson', 'spearman', 'kendall'):
            c = sample_corr['point'][method]
            print(f"   {method.capitalize():<9}: "
                  f"P={c['precision']:.4f}, R={c['recall']:.4f}, F1={c['f1']:.4f}")

    # ---- Text-level: per-MR correlation, averaged across MRs ----
    # For each MR, correlate model P/R/F1 vs human quality across available systems.
    # Requires >= 2 systems per MR to compute a correlation.
    mr_ids = [mid for mid, d in mr_data.items() if len(d['prec']) >= 2]

    def _text_level_point(mr_ids_subset):
        """Compute mean per-MR correlation over a given subset of MR ids."""
        pearson_p, pearson_r, pearson_f = [], [], []
        spear_p,   spear_r,   spear_f   = [], [], []
        kend_p,    kend_r,    kend_f    = [], [], []
        for mid in mr_ids_subset:
            d = mr_data[mid]
            x_p = np.array(d['prec'])
            x_r = np.array(d['rec'])
            x_f = np.array(d['f1'])
            y   = np.array(d['qual'])
            if len(y) < 2:
                continue
            pearson_p.append(_safe_corr(pearsonr, x_p, y));  pearson_r.append(_safe_corr(pearsonr, x_r, y));  pearson_f.append(_safe_corr(pearsonr, x_f, y))
            spear_p.append(_safe_corr(spearmanr, x_p, y));   spear_r.append(_safe_corr(spearmanr, x_r, y));   spear_f.append(_safe_corr(spearmanr, x_f, y))
            kend_p.append(_safe_corr(kendalltau, x_p, y));   kend_r.append(_safe_corr(kendalltau, x_r, y));   kend_f.append(_safe_corr(kendalltau, x_f, y))
        return {
            'pearson':  {'precision': _safe_mean(pearson_p), 'recall': _safe_mean(pearson_r), 'f1': _safe_mean(pearson_f)},
            'spearman': {'precision': _safe_mean(spear_p),   'recall': _safe_mean(spear_r),   'f1': _safe_mean(spear_f)},
            'kendall':  {'precision': _safe_mean(kend_p),    'recall': _safe_mean(kend_r),    'f1': _safe_mean(kend_f)},
        }

    text_corr = {}
    if mr_ids:
        point = _text_level_point(mr_ids)

        # Bootstrap over MRs — keep ALL iterations, no p-value filtering
        boot_pearson_p, boot_pearson_r, boot_pearson_f = [], [], []
        boot_spear_p,   boot_spear_r,   boot_spear_f   = [], [], []
        boot_kend_p,    boot_kend_r,    boot_kend_f    = [], [], []
        for _ in range(n_bootstrap):
            sample_ids = random.choices(mr_ids, k=len(mr_ids))
            b = _text_level_point(sample_ids)
            boot_pearson_p.append(b['pearson']['precision']);  boot_pearson_r.append(b['pearson']['recall']);  boot_pearson_f.append(b['pearson']['f1'])
            boot_spear_p.append(b['spearman']['precision']);   boot_spear_r.append(b['spearman']['recall']);   boot_spear_f.append(b['spearman']['f1'])
            boot_kend_p.append(b['kendall']['precision']);     boot_kend_r.append(b['kendall']['recall']);     boot_kend_f.append(b['kendall']['f1'])

        text_corr = {
            'n_mr': len(mr_ids),
            'point': point,
            'bootstrap': {
                'pearson':  {'precision': _bootstrap_ci(boot_pearson_p), 'recall': _bootstrap_ci(boot_pearson_r), 'f1': _bootstrap_ci(boot_pearson_f)},
                'spearman': {'precision': _bootstrap_ci(boot_spear_p),   'recall': _bootstrap_ci(boot_spear_r),   'f1': _bootstrap_ci(boot_spear_f)},
                'kendall':  {'precision': _bootstrap_ci(boot_kend_p),    'recall': _bootstrap_ci(boot_kend_r),    'f1': _bootstrap_ci(boot_kend_f)},
            }
        }
        print(f"📊 Text-level ({len(mr_ids)} MRs, {n_bootstrap} bootstrap):")
        for method in ('pearson', 'spearman', 'kendall'):
            c = point[method]
            print(f"   {method.capitalize():<9}: "
                  f"P={c['precision']:.4f}, R={c['recall']:.4f}, F1={c['f1']:.4f}")

    # ---- System-level: one data point per system ----
    sys_corr = {}
    sys_names = sorted(sys_model.keys())
    if len(sys_names) > 1:
        sys_prec = np.array([np.mean(sys_model[s]['prec']) for s in sys_names])
        sys_rec  = np.array([np.mean(sys_model[s]['rec'])  for s in sys_names])
        sys_f1   = np.array([np.mean(sys_model[s]['f1'])   for s in sys_names])
        sys_qual = np.array([np.mean(sys_human[s])         for s in sys_names])

        sys_point = {
            'pearson':  {'precision': _safe_corr(pearsonr, sys_prec, sys_qual),
                         'recall':    _safe_corr(pearsonr, sys_rec,  sys_qual),
                         'f1':        _safe_corr(pearsonr, sys_f1,   sys_qual)},
            'spearman': {'precision': _safe_corr(spearmanr, sys_prec, sys_qual),
                         'recall':    _safe_corr(spearmanr, sys_rec,  sys_qual),
                         'f1':        _safe_corr(spearmanr, sys_f1,   sys_qual)},
            'kendall':  {'precision': _safe_corr(kendalltau, sys_prec, sys_qual),
                         'recall':    _safe_corr(kendalltau, sys_rec,  sys_qual),
                         'f1':        _safe_corr(kendalltau, sys_f1,   sys_qual)},
        }

        boot_pear_p, boot_pear_r, boot_pear_f = [], [], []
        boot_spear_p, boot_spear_r, boot_spear_f = [], [], []
        boot_kend_p, boot_kend_r, boot_kend_f = [], [], []
        for _ in range(n_bootstrap):
            sampled_systems = random.choices(sys_names, k=len(sys_names))
            b_prec = np.array([np.mean(sys_model[s]['prec']) for s in sampled_systems])
            b_rec = np.array([np.mean(sys_model[s]['rec']) for s in sampled_systems])
            b_f1 = np.array([np.mean(sys_model[s]['f1']) for s in sampled_systems])
            b_qual = np.array([np.mean(sys_human[s]) for s in sampled_systems])

            boot_pear_p.append(_safe_corr(pearsonr, b_prec, b_qual))
            boot_pear_r.append(_safe_corr(pearsonr, b_rec, b_qual))
            boot_pear_f.append(_safe_corr(pearsonr, b_f1, b_qual))

            boot_spear_p.append(_safe_corr(spearmanr, b_prec, b_qual))
            boot_spear_r.append(_safe_corr(spearmanr, b_rec, b_qual))
            boot_spear_f.append(_safe_corr(spearmanr, b_f1, b_qual))

            boot_kend_p.append(_safe_corr(kendalltau, b_prec, b_qual))
            boot_kend_r.append(_safe_corr(kendalltau, b_rec, b_qual))
            boot_kend_f.append(_safe_corr(kendalltau, b_f1, b_qual))

        sys_corr = {
            'n_systems': len(sys_names),
            'systems': sys_names,
            'point': sys_point,
            'bootstrap': {
                'pearson':  {'precision': _bootstrap_ci(boot_pear_p), 'recall': _bootstrap_ci(boot_pear_r), 'f1': _bootstrap_ci(boot_pear_f)},
                'spearman': {'precision': _bootstrap_ci(boot_spear_p), 'recall': _bootstrap_ci(boot_spear_r), 'f1': _bootstrap_ci(boot_spear_f)},
                'kendall':  {'precision': _bootstrap_ci(boot_kend_p), 'recall': _bootstrap_ci(boot_kend_r), 'f1': _bootstrap_ci(boot_kend_f)},
            },
        }
        print(f"📊 System-level ({len(sys_names)} systems):")
        for method in ('pearson', 'spearman', 'kendall'):
            c = sys_point[method]
            print(f"   {method.capitalize():<9}: "
                  f"P={c['precision']:.4f}, R={c['recall']:.4f}, F1={c['f1']:.4f}")

    return {
        'model': model_name,
        'sample_level': sample_corr,
        'text_level': text_corr,
        'system_level': sys_corr,
    }


# Main

def main():
    print("LLM EVALUATION - E2E HUMAN RATINGS")

    set_random_seed(RANDOM_SEED)
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    run_tag = datetime.now().strftime('%Y%m%d_%H%M%S')
    visible_gpu_count = infer_visible_gpu_count(GPU_ID)
    print(f"🖥️  Visible GPUs: {visible_gpu_count} (CUDA_VISIBLE_DEVICES={GPU_ID})")

    # Load and flatten data
    print(f"\n📂 Loading: {INPUT_JSON}")
    if not INPUT_JSON.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_JSON}")
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        converted = json.load(f)
    data = flatten_data(converted)
    print(f"📊 Total (MR, sys) pairs: {len(data)}")

    all_correlations = []

    for idx, checkpoint_config in enumerate(CHECKPOINTS, 1):
        print(f"\n[{idx}/{len(CHECKPOINTS)}] {checkpoint_config['name']}")
        batch_size = resolve_batch_size(checkpoint_config, visible_gpu_count)
        print(f"📦 Effective global batch size: {batch_size}")

        results_file     = Path(OUTPUT_DIR) / f"{checkpoint_config['name']}_results.json"
        correlation_file = Path(OUTPUT_DIR) / f"{checkpoint_config['name']}_correlation.json"

        engine = None
        try:
            # Load model
            print("🔧 Loading model...")
            engine, request_config = load_model(checkpoint_config, batch_size)
            print("✅ Model loaded!")

            # Run inference
            print("🚀 Running evaluation...")
            results = run_evaluation(data, engine, request_config, batch_size)

            # Save results immediately - don't wait for correlation
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"💾 Results saved: {results_file}")

            # Compute correlations from saved results
            correlation = compute_correlations(results, checkpoint_config['name'])
            all_correlations.append(correlation)
            with open(correlation_file, 'w', encoding='utf-8') as f:
                json.dump(correlation, f, indent=2, ensure_ascii=False)
            print(f"💾 Correlation saved: {correlation_file}")

        except Exception as e:
            print(f"❌ Error during evaluation: {e}")
            traceback.print_exc()
            raise
        finally:
            if engine is not None:
                del engine
            _clear_cuda_memory()

    # Save summary
    summary_file = Path(OUTPUT_DIR) / f"summary_llama_{run_tag}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_correlations, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Summary saved: {summary_file}")

    print("\n✅ EVALUATION COMPLETE!")


if __name__ == '__main__':
    main()
