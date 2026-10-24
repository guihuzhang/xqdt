#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate prediction results from JSON files
Calculate instance-level and global-level metrics (triple-level, element-level, similarity)
"""
import json
import re
import sys
import numpy as np
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from sentence_transformers import SentenceTransformer
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from verifier_train_eval_e2e19.path_config import (
    PROMPT_BASELINES_DIR,
    PROMPT_BASELINE_EVAL_DIR,
    PROMPT_BASELINE_EVAL_LOG_DIR,
    ensure_dir,
)


class Logger:
    """Write logs to both terminal and file."""

    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'a', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

    def __enter__(self):
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.terminal
        self.log.close()
        return False  # Do not swallow exceptions


def triple_to_str(triple: Tuple[str, str, str]) -> str:
    """Convert tuple to string"""
    return ' | '.join(triple)


def parse_triple(triple_str: str) -> Optional[Tuple[str, str, str]]:
    """Parse [S] ... [P] ... [O] ... format"""
    pattern = r'\[S\]\s*(.+?)\s*\[P\]\s*(.+?)\s*\[O\]\s*(.+?)(?:\s*\||$)'
    match = re.search(pattern, triple_str.strip(), re.DOTALL)
    if match:
        return match.group(1).strip("* "), match.group(2).strip("* "), match.group(3).strip("* ")
    return None


def compute_triple_level_metrics(predicted: List[Tuple], gold: List[Tuple]) -> Dict:
    """Compute triple-level precision, recall, F1 using Hungarian algorithm"""

    if not predicted and not gold:
        return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'tp': 0, 'fp': 0, 'fn': 0}

    if not predicted:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'tp': 0, 'fp': 0, 'fn': len(gold)}

    if not gold:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'tp': 0, 'fp': len(predicted), 'fn': 0}

    n_pred = len(predicted)
    n_gold = len(gold)

    # Build cost matrix: cost = 0 if exact match, 1 otherwise
    cost_matrix = np.ones((n_pred, n_gold))

    for i, pred_triple in enumerate(predicted):
        pred_str = triple_to_str(pred_triple).lower()
        for j, gold_triple in enumerate(gold):
            gold_str = triple_to_str(gold_triple).lower()
            if pred_str == gold_str:
                cost_matrix[i][j] = 0  # Exact match, cost = 0

    # Apply Hungarian algorithm
    pred_indices, gold_indices = linear_sum_assignment(cost_matrix)

    # Count true positives (exact matches)
    tp = 0
    matched_pred = set()
    matched_gold = set()

    for pred_idx, gold_idx in zip(pred_indices, gold_indices):
        if cost_matrix[pred_idx, gold_idx] == 0:  # Exact match
            tp += 1
            matched_pred.add(pred_idx)
            matched_gold.add(gold_idx)

    # Count false positives and false negatives
    fp = n_pred - len(matched_pred)
    fn = n_gold - len(matched_gold)

    # Calculate metrics
    precision = tp / n_pred if n_pred > 0 else 0.0
    recall = tp / n_gold if n_gold > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }


def compute_element_level_metrics(predicted: List[Tuple], gold: List[Tuple]) -> Dict:
    """Compute element-level precision, recall, F1 using Hungarian algorithm"""

    if not predicted or not gold:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'matched_elements': 0,
            'total_pred_elements': len(predicted) * 3,
            'total_gold_elements': len(gold) * 3,
            'num_pairs': 0
        }

    # Build cost matrix: cost_matrix[i][j] = negative number of matched elements
    n_pred = len(predicted)
    n_gold = len(gold)
    cost_matrix = np.zeros((n_pred, n_gold))

    for i, pred_triple in enumerate(predicted):
        for j, gold_triple in enumerate(gold):
            # Count matched elements in this pair
            matched = 0
            for pred_elem, gold_elem in zip(pred_triple, gold_triple):
                if pred_elem.strip().lower() == gold_elem.strip().lower():
                    matched += 1
            # Negative because we minimize cost
            cost_matrix[i][j] = -matched

    # Hungarian algorithm to find optimal matching
    pred_indices, gold_indices = linear_sum_assignment(cost_matrix)

    # Count total matched elements
    matched_elements = 0
    for i, j in zip(pred_indices, gold_indices):
        matched_elements += int(-cost_matrix[i][j])

    total_pred_elements = len(predicted) * 3
    precision = matched_elements / total_pred_elements if total_pred_elements > 0 else 0.0

    total_gold_elements = len(gold) * 3
    recall = matched_elements / total_gold_elements if total_gold_elements > 0 else 0.0

    # F1
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'matched_elements': int(matched_elements),
        'total_pred_elements': total_pred_elements,
        'total_gold_elements': total_gold_elements,
        'num_pairs': len(pred_indices),
        'avg_match_per_pair': matched_elements / len(pred_indices) if len(pred_indices) > 0 else 0.0
    }


def compute_similarity_metrics(predicted: List[Tuple], gold: List[Tuple], embedder=None) -> Dict:
    """Compute similarity using BGE embeddings with Hungarian algorithm for optimal matching"""
    if embedder is None or not predicted or not gold:
        return {
            'avg_similarity': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'num_matches': 0
        }

    pred_texts = [triple_to_str(t) for t in predicted]
    gold_texts = [triple_to_str(t) for t in gold]

    try:
        pred_embeds = embedder.encode(pred_texts, normalize_embeddings=True)
        gold_embeds = embedder.encode(gold_texts, normalize_embeddings=True)

        # Build similarity matrix
        n_pred = len(predicted)
        n_gold = len(gold)
        similarity_matrix = np.zeros((n_pred, n_gold))

        for i, pred_emb in enumerate(pred_embeds):
            for j, gold_emb in enumerate(gold_embeds):
                similarity_matrix[i][j] = np.dot(pred_emb, gold_emb)

        # Use Hungarian algorithm to find optimal matching
        cost_matrix = -similarity_matrix
        pred_indices, gold_indices = linear_sum_assignment(cost_matrix)

        # Sum of similarities = "soft" true positives
        total_similarity = 0.0
        for i, j in zip(pred_indices, gold_indices):
            total_similarity += similarity_matrix[i][j]

        # Precision: average match quality per prediction
        precision = total_similarity / n_pred if n_pred > 0 else 0.0

        # Recall: average match quality per gold triple
        recall = total_similarity / n_gold if n_gold > 0 else 0.0

        # F1
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        avg_sim = total_similarity / len(pred_indices) if len(pred_indices) > 0 else 0.0

        return {
            'avg_similarity': float(avg_sim),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'num_matches': len(pred_indices)
        }
    except Exception as e:
        print(f"⚠️ Similarity computation failed: {e}")
        return {
            'avg_similarity': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'num_matches': 0
        }


def extract_gold_triples(sample: Dict) -> Dict:
    """Extract gold standard triples from sample"""
    gold = {'missing': [], 'extra': [], 'incorrect': []}

    # Missing triples
    for triple_str in sample.get('gt_missing_triples', []):
        parts = [p.strip() for p in triple_str.split('|')]
        if len(parts) == 3:
            gold['missing'].append(tuple(parts))

    # Extra triples
    for triple_str in sample.get('gt_extra_triples', []):
        parts = [p.strip() for p in triple_str.split('|')]
        if len(parts) == 3:
            gold['extra'].append(tuple(parts))

    # Incorrect triples
    for mapping in sample.get('gt_incorrect_mappings', []):
        incorrect_str = mapping.get('incorrect', '')
        parts = [p.strip() for p in incorrect_str.split('|')]
        if len(parts) == 3:
            gold['incorrect'].append(tuple(parts))

    return gold


def extract_predicted_triples(sample: Dict) -> Dict:
    """Extract predicted triples from parsed_output"""
    predicted = {'missing': [], 'extra': [], 'incorrect': []}

    # Get parsed_output
    parsed = sample.get('parsed_output', {})

    # Parse each triple string using parse_triple()
    for error_type in ['missing', 'extra', 'incorrect']:
        for triple_str in parsed.get(error_type, []):
            triple_tuple = parse_triple(triple_str)
            if triple_tuple:
                predicted[error_type].append(triple_tuple)

    return predicted


def evaluate_sample(predicted: Dict, gold: Dict, embedder=None, is_positive: bool = False) -> Dict:
    """
    Evaluate a single sample
    Computes triple-level, element-level, and similarity metrics
    """
    results = {}

    for error_type in ['missing', 'extra', 'incorrect']:
        pred = predicted.get(error_type, [])
        gold_list = gold.get(error_type, [])

        # Special handling for positive samples (no errors expected)
        if is_positive:
            if len(pred) == 0:
                # Correct: no errors predicted
                triple_metrics = {
                    'precision': 1.0,
                    'recall': 1.0,
                    'f1': 1.0,
                    'tp': 0,
                    'fp': 0,
                    'fn': 0
                }
                element_metrics = {
                    'precision': 1.0,
                    'recall': 1.0,
                    'f1': 1.0,
                    'matched_elements': 0,
                    'total_pred_elements': 0,
                    'total_gold_elements': 0,
                    'num_pairs': 0
                }
                sim_metrics = {
                    'avg_similarity': 1.0,
                    'precision': 1.0,
                    'recall': 1.0,
                    'f1': 1.0
                }
            else:
                # Incorrect: false positives
                triple_metrics = {
                    'precision': 0.0,
                    'recall': 1.0,
                    'f1': 0.0,
                    'tp': 0,
                    'fp': len(pred),
                    'fn': 0
                }
                element_metrics = {
                    'precision': 0.0,
                    'recall': 1.0,
                    'f1': 0.0,
                    'matched_elements': 0,
                    'total_pred_elements': len(pred) * 3,
                    'total_gold_elements': 0,
                    'num_pairs': 0
                }
                sim_metrics = {
                    'avg_similarity': 0.0,
                    'precision': 0.0,
                    'recall': 1.0,
                    'f1': 0.0
                }
        else:
            # Normal computation for negative samples
            triple_metrics = compute_triple_level_metrics(pred, gold_list)
            element_metrics = compute_element_level_metrics(pred, gold_list)
            sim_metrics = compute_similarity_metrics(pred, gold_list, embedder)

        results[error_type] = {
            'triple_level': triple_metrics,
            'element_level': element_metrics,
            'similarity': sim_metrics,
            'predicted_count': len(pred),
            'gold_count': len(gold_list)
        }

    return results


def compute_global_metrics(all_metrics: List[Dict]) -> Dict:
    """
    Compute global metrics by summing TP/FP/FN across all samples
    Then calculate precision, recall, F1 from the totals
    Includes both Macro and Micro averaging for all levels
    """

    # Initialize accumulators for each error type
    global_totals = {
        'missing': {
            'triple_level': {'tp': 0, 'fp': 0, 'fn': 0},
            'element_level': {'matched': 0, 'pred_total': 0, 'gold_total': 0},
            'similarity': {'total_sim': 0.0, 'pred_count': 0, 'gold_count': 0}
        },
        'extra': {
            'triple_level': {'tp': 0, 'fp': 0, 'fn': 0},
            'element_level': {'matched': 0, 'pred_total': 0, 'gold_total': 0},
            'similarity': {'total_sim': 0.0, 'pred_count': 0, 'gold_count': 0}
        },
        'incorrect': {
            'triple_level': {'tp': 0, 'fp': 0, 'fn': 0},
            'element_level': {'matched': 0, 'pred_total': 0, 'gold_total': 0},
            'similarity': {'total_sim': 0.0, 'pred_count': 0, 'gold_count': 0}
        }
    }

    # Accumulate counts from all samples
    for sample_metrics in all_metrics:
        for error_type in ['missing', 'extra', 'incorrect']:
            if error_type in sample_metrics:
                # Triple-level: sum TP/FP/FN
                tl = sample_metrics[error_type]['triple_level']
                global_totals[error_type]['triple_level']['tp'] += tl['tp']
                global_totals[error_type]['triple_level']['fp'] += tl['fp']
                global_totals[error_type]['triple_level']['fn'] += tl['fn']

                # Element-level: sum matched elements and totals
                el = sample_metrics[error_type]['element_level']
                global_totals[error_type]['element_level']['matched'] += el['matched_elements']
                global_totals[error_type]['element_level']['pred_total'] += el['total_pred_elements']
                global_totals[error_type]['element_level']['gold_total'] += el['total_gold_elements']

                # Similarity: accumulate for P/R/F1 calculation
                sm = sample_metrics[error_type]['similarity']
                pred_count = sample_metrics[error_type]['predicted_count']
                gold_count = sample_metrics[error_type]['gold_count']

                # Total similarity score from this sample
                # avg_similarity * num_matches gives total_similarity
                num_matches = sm.get('num_matches', 0)
                if num_matches > 0:
                    total_similarity = sm['avg_similarity'] * num_matches
                    global_totals[error_type]['similarity']['total_sim'] += total_similarity

                global_totals[error_type]['similarity']['pred_count'] += pred_count
                global_totals[error_type]['similarity']['gold_count'] += gold_count

    # Compute per-class metrics from totals
    result = {}
    macro_triple_precision_vals = []
    macro_triple_recall_vals = []
    macro_triple_f1_vals = []
    macro_element_precision_vals = []
    macro_element_recall_vals = []
    macro_element_f1_vals = []
    macro_sim_precision_vals = []
    macro_sim_recall_vals = []
    macro_sim_f1_vals = []
    for error_type in ['missing', 'extra', 'incorrect']:
        totals = global_totals[error_type]

        # Triple-level P/R/F1 from summed TP/FP/FN
        tp = totals['triple_level']['tp']
        fp = totals['triple_level']['fp']
        fn = totals['triple_level']['fn']

        triple_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        triple_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        triple_f1 = 2 * triple_precision * triple_recall / (triple_precision + triple_recall) if (
                                                                                                         triple_precision + triple_recall) > 0 else 0.0

        # Element-level P/R/F1 from summed matched/totals
        matched = totals['element_level']['matched']
        pred_total = totals['element_level']['pred_total']
        gold_total = totals['element_level']['gold_total']

        element_precision = matched / pred_total if pred_total > 0 else 0.0
        element_recall = matched / gold_total if gold_total > 0 else 0.0
        element_f1 = 2 * element_precision * element_recall / (element_precision + element_recall) if (
                                                                                                              element_precision + element_recall) > 0 else 0.0

        # Similarity P/R/F1
        total_sim = totals['similarity']['total_sim']
        sim_pred_count = totals['similarity']['pred_count']
        sim_gold_count = totals['similarity']['gold_count']

        sim_precision = total_sim / sim_pred_count if sim_pred_count > 0 else 0.0
        sim_recall = total_sim / sim_gold_count if sim_gold_count > 0 else 0.0
        sim_f1 = 2 * sim_precision * sim_recall / (sim_precision + sim_recall) if (
                                                                                          sim_precision + sim_recall) > 0 else 0.0

        macro_triple_precision_vals.append(triple_precision)
        macro_triple_recall_vals.append(triple_recall)
        macro_triple_f1_vals.append(triple_f1)
        macro_element_precision_vals.append(element_precision)
        macro_element_recall_vals.append(element_recall)
        macro_element_f1_vals.append(element_f1)
        macro_sim_precision_vals.append(sim_precision)
        macro_sim_recall_vals.append(sim_recall)
        macro_sim_f1_vals.append(sim_f1)

        result[error_type] = {
            'triple_level': {
                'precision': triple_precision,
                'recall': triple_recall,
                'f1': triple_f1,
                'tp': tp,
                'fp': fp,
                'fn': fn
            },
            'element_level': {
                'precision': element_precision,
                'recall': element_recall,
                'f1': element_f1,
                'matched_elements': matched,
                'total_pred_elements': pred_total,
                'total_gold_elements': gold_total
            },
            'similarity': {
                'precision': sim_precision,
                'recall': sim_recall,
                'f1': sim_f1,
                'total_similarity': total_sim
            },
            'num_samples': len(all_metrics)
        }

    # ============= MACRO AVERAGE =============
    macro_triple_precision = np.mean(macro_triple_precision_vals)
    macro_triple_recall = np.mean(macro_triple_recall_vals)
    macro_triple_f1 = np.mean(macro_triple_f1_vals)

    macro_element_precision = np.mean(macro_element_precision_vals)
    macro_element_recall = np.mean(macro_element_recall_vals)
    macro_element_f1 = np.mean(macro_element_f1_vals)

    macro_sim_precision = np.mean(macro_sim_precision_vals)
    macro_sim_recall = np.mean(macro_sim_recall_vals)
    macro_sim_f1 = np.mean(macro_sim_f1_vals)

    # ============= MICRO AVERAGE =============
    # Triple-level micro
    total_triple_tp = sum([result[et]['triple_level']['tp'] for et in ['missing', 'extra', 'incorrect']])
    total_triple_fp = sum([result[et]['triple_level']['fp'] for et in ['missing', 'extra', 'incorrect']])
    total_triple_fn = sum([result[et]['triple_level']['fn'] for et in ['missing', 'extra', 'incorrect']])

    micro_triple_precision = total_triple_tp / (total_triple_tp + total_triple_fp) if (
                                                                                              total_triple_tp + total_triple_fp) > 0 else 0.0
    micro_triple_recall = total_triple_tp / (total_triple_tp + total_triple_fn) if (
                                                                                           total_triple_tp + total_triple_fn) > 0 else 0.0
    micro_triple_f1 = 2 * micro_triple_precision * micro_triple_recall / (
            micro_triple_precision + micro_triple_recall) if (
                                                                     micro_triple_precision + micro_triple_recall) > 0 else 0.0

    # Element-level micro
    total_element_matched = sum(
        [result[et]['element_level']['matched_elements'] for et in ['missing', 'extra', 'incorrect']])
    total_element_pred = sum(
        [result[et]['element_level']['total_pred_elements'] for et in ['missing', 'extra', 'incorrect']])
    total_element_gold = sum(
        [result[et]['element_level']['total_gold_elements'] for et in ['missing', 'extra', 'incorrect']])

    micro_element_precision = total_element_matched / total_element_pred if total_element_pred > 0 else 0.0
    micro_element_recall = total_element_matched / total_element_gold if total_element_gold > 0 else 0.0
    micro_element_f1 = 2 * micro_element_precision * micro_element_recall / (
            micro_element_precision + micro_element_recall) if (
                                                                       micro_element_precision + micro_element_recall) > 0 else 0.0

    # Similarity micro
    total_sim = sum([result[et]['similarity']['total_similarity'] for et in ['missing', 'extra', 'incorrect']])
    total_sim_pred = sum([global_totals[et]['similarity']['pred_count'] for et in ['missing', 'extra', 'incorrect']])
    total_sim_gold = sum([global_totals[et]['similarity']['gold_count'] for et in ['missing', 'extra', 'incorrect']])

    micro_sim_precision = total_sim / total_sim_pred if total_sim_pred > 0 else 0.0
    micro_sim_recall = total_sim / total_sim_gold if total_sim_gold > 0 else 0.0
    micro_sim_f1 = 2 * micro_sim_precision * micro_sim_recall / (
            micro_sim_precision + micro_sim_recall) if (micro_sim_precision + micro_sim_recall) > 0 else 0.0

    # Store both macro and micro averages
    result['macro_average'] = {
        'triple_level': {
            'precision': macro_triple_precision,
            'recall': macro_triple_recall,
            'f1': macro_triple_f1
        },
        'element_level': {
            'precision': macro_element_precision,
            'recall': macro_element_recall,
            'f1': macro_element_f1
        },
        'similarity': {
            'precision': macro_sim_precision,
            'recall': macro_sim_recall,
            'f1': macro_sim_f1
        }
    }

    result['micro_average'] = {
        'triple_level': {
            'precision': micro_triple_precision,
            'recall': micro_triple_recall,
            'f1': micro_triple_f1,
            'tp': total_triple_tp,
            'fp': total_triple_fp,
            'fn': total_triple_fn
        },
        'element_level': {
            'precision': micro_element_precision,
            'recall': micro_element_recall,
            'f1': micro_element_f1,
            'matched_elements': total_element_matched,
            'total_pred_elements': total_element_pred,
            'total_gold_elements': total_element_gold
        },
        'similarity': {
            'precision': micro_sim_precision,
            'recall': micro_sim_recall,
            'f1': micro_sim_f1,
            'total_similarity': total_sim
        }
    }

    return result


# Keep evaluate_predictions print section consistent with global metrics
def evaluate_predictions(input_json: str, output_json: str = None, use_embedder: bool = False,
                         embedder: Optional[SentenceTransformer] = None):
    """
    Evaluate predictions from JSON file
    """
    print(f"\nEvaluating: {input_json}")

    # Load predictions
    with open(input_json, 'r', encoding='utf-8') as f:
        samples = json.load(f)

    print(f"Total samples: {len(samples)}")

    # Initialize embedder if needed
    local_embedder = embedder
    if local_embedder is not None:
        print("Using preloaded embedder for similarity computation")
    elif use_embedder:
        try:
            local_embedder = SentenceTransformer('BAAI/bge-large-en-v1.5')
            print("Loaded embedder for similarity computation")
        except Exception:
            print("⚠️ Failed to load embedder, skipping similarity metrics")

    # Evaluate each sample
    all_metrics = []
    evaluated_samples = []

    for sample in tqdm(samples):
        is_positive = sample.get('label') == 'positive'

        # Extract predicted and gold triples
        predicted = extract_predicted_triples(sample)
        gold = extract_gold_triples(sample)

        # Compute metrics
        sample_metrics = evaluate_sample(predicted, gold, local_embedder, is_positive)
        all_metrics.append(sample_metrics)

        # Add metrics to sample
        sample_with_metrics = sample.copy()
        sample_with_metrics['evaluation_metrics'] = sample_metrics
        evaluated_samples.append(sample_with_metrics)

    # Compute global metrics
    global_metrics = compute_global_metrics(all_metrics)

    # Print results
    print("\nGlobal Metrics:")
    print("=" * 80)

    for error_type in ['missing', 'extra', 'incorrect']:
        m = global_metrics[error_type]
        print(f"\n{error_type.upper()}:")

        # Triple-level
        tl = m['triple_level']
        print(f"  Triple-level  -> P: {tl['precision']:.4f}, R: {tl['recall']:.4f}, F1: {tl['f1']:.4f}")

        # Element-level
        el = m['element_level']
        print(f"  Element-level -> P: {el['precision']:.4f}, R: {el['recall']:.4f}, F1: {el['f1']:.4f}")

        # Similarity
        if local_embedder:
            sim = m['similarity']
            print(f"  Similarity    -> P: {sim['precision']:.4f}, R: {sim['recall']:.4f}, F1: {sim['f1']:.4f}")

        print(f"  Total samples: {m['num_samples']}")

    print("\nMACRO AVERAGE (average of per-class metrics):")
    macro = global_metrics['macro_average']
    print(
        f"  Triple-level  -> P: {macro['triple_level']['precision']:.4f}, R: {macro['triple_level']['recall']:.4f}, F1: {macro['triple_level']['f1']:.4f}")
    print(
        f"  Element-level -> P: {macro['element_level']['precision']:.4f}, R: {macro['element_level']['recall']:.4f}, F1: {macro['element_level']['f1']:.4f}")
    if local_embedder:
        print(
            f"  Similarity    -> P: {macro['similarity']['precision']:.4f}, R: {macro['similarity']['recall']:.4f}, F1: {macro['similarity']['f1']:.4f}")

    print(f"\nMICRO AVERAGE (computed from summed TP/FP/FN):")
    micro = global_metrics['micro_average']
    print(
        f"  Triple-level  -> P: {micro['triple_level']['precision']:.4f}, R: {micro['triple_level']['recall']:.4f}, F1: {micro['triple_level']['f1']:.4f}")
    print(
        f"  Element-level -> P: {micro['element_level']['precision']:.4f}, R: {micro['element_level']['recall']:.4f}, F1: {micro['element_level']['f1']:.4f}")
    if local_embedder:
        print(
            f"  Similarity    -> P: {micro['similarity']['precision']:.4f}, R: {micro['similarity']['recall']:.4f}, F1: {micro['similarity']['f1']:.4f}")

    # Save results
    if output_json is None:
        output_json = input_json.replace('.json', '_evaluated.json')

    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        'input_file': str(input_json),
        'total_samples': len(samples),
        'global_metrics': global_metrics,
        'samples': evaluated_samples
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_path}")

    return results


def evaluate_multiple_models(json_files: List[str], use_embedder: bool = False):
    """Compare multiple model outputs"""

    print("\nComparing Multiple Models")
    print("=" * 100)

    all_results = {}
    shared_embedder = None
    if use_embedder:
        try:
            shared_embedder = SentenceTransformer('BAAI/bge-large-en-v1.5')
            print("Loaded shared embedder for all models")
        except Exception:
            print("⚠️ Failed to load shared embedder, skipping similarity metrics")
            use_embedder = False

    for json_file in json_files:
        model_name = Path(json_file).stem
        results = evaluate_predictions(
            json_file,
            output_json=None,
            use_embedder=use_embedder,
            embedder=shared_embedder)
        all_results[model_name] = results['global_metrics']

    # Print comparison table
    print("\n\nComparison Table (Triple-level):")
    print("-" * 100)
    print(f"{'Model':<40} {'Error Type':<15} {'P':<8} {'R':<8} {'F1':<8}")
    print("-" * 100)

    for model_name, metrics in all_results.items():
        for error_type in ['missing', 'extra', 'incorrect']:
            m = metrics[error_type]['triple_level']
            print(f"{model_name:<40} {error_type:<15} {m['precision']:<8.4f} {m['recall']:<8.4f} {m['f1']:<8.4f}")
        print()


if __name__ == '__main__':
    log_dir = ensure_dir(PROMPT_BASELINE_EVAL_LOG_DIR)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'evaluation_{timestamp}.log'

    models_to_eval = [
        "gpt-4.1_openrouter",
        "gpt-5.1_openrouter",
        "gemma3_27b-it-q8_0_ollama",
        "llama3_3_70b-instruct-q8_0_ollama",
        "qwen3_nothink_32b_ollama",
    ]

    with Logger(log_file) as logger:
        print(f"Evaluation Log")
        print(f"Started at: {datetime.now()}")
        print(f"Log file: {log_file}")

        shared_embedder = None
        try:
            shared_embedder = SentenceTransformer('BAAI/bge-large-en-v1.5')
            print("Loaded shared embedder for this evaluation run")
        except Exception:
            print("⚠️ Failed to load shared embedder, similarity metrics will be skipped")

        for i, model in enumerate(models_to_eval, 1):
            print(f"\n{'=' * 80}")
            print(f"Model {i}/{len(models_to_eval)}: {model}")
            print(f"{'=' * 80}")

            evaluate_predictions(
                input_json=str(PROMPT_BASELINES_DIR / f'test_predictions_{model}.json'),
                output_json=str(ensure_dir(PROMPT_BASELINE_EVAL_DIR) / f'test_predictions_{model}.json'),
                use_embedder=shared_embedder is not None,
                embedder=shared_embedder
            )

        print(f"All evaluations completed at: {datetime.now()}")
