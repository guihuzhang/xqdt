"""
Inter-Annotator Agreement for Annotation Process
Calculates agreement on:
1. Part 1: Judgments on model predictions (categorical) - TWO VERSIONS
   - Version A: 4 categories (over_prediction, correct, wrong_type, wrong_triple)
   - Version B: 3 categories (over_prediction, correct, wrong - merged last two)
2. Part 2: Counts of supplemented errors (numerical)
"""

import json
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
from pathlib import Path


BASE = Path(__file__).resolve().parent


# ============================================================================
# KRIPPENDORFF'S ALPHA
# ============================================================================

def krippendorff_alpha(data_matrix: np.ndarray, level_of_measurement='nominal') -> float:
    """
    Calculate Krippendorff's Alpha for multi-rater agreement

    Args:
        data_matrix: Matrix of shape (n_raters, n_items)
                    Use np.nan for missing values
        level_of_measurement: 'nominal', 'ordinal', 'interval', or 'ratio'

    Returns:
        Krippendorff's Alpha value (-1 to 1)
    """

    data = data_matrix.copy()
    n_raters, n_items = data.shape

    # Get unique values (excluding NaN)
    values = np.unique(data[~np.isnan(data)])
    n_values = len(values)

    if n_values == 0:
        return np.nan

    # Create value to index mapping
    value_to_idx = {v: i for i, v in enumerate(values)}

    # Initialize coincidence matrix
    coincidence_matrix = np.zeros((n_values, n_values))

    # Fill coincidence matrix
    for item_idx in range(n_items):
        item_values = data[:, item_idx]
        valid_values = item_values[~np.isnan(item_values)]

        if len(valid_values) < 2:
            continue

        # Weight for this item
        m_u = len(valid_values)
        weight = 1.0 / (m_u - 1)

        # Add pairwise coincidences
        for i in range(len(valid_values)):
            for j in range(len(valid_values)):
                if i != j:
                    v_i = valid_values[i]
                    v_j = valid_values[j]
                    idx_i = value_to_idx[v_i]
                    idx_j = value_to_idx[v_j]
                    coincidence_matrix[idx_i, idx_j] += weight

    # Calculate observed disagreement
    n_c = np.sum(coincidence_matrix)

    if n_c == 0:
        return np.nan

    # Define delta function based on level of measurement
    if level_of_measurement == 'nominal':
        # For nominal: delta = 0 if same, 1 if different
        delta = 1 - np.eye(n_values)
    else:
        # For interval/ratio: delta = (v_i - v_j)^2
        delta = np.zeros((n_values, n_values))
        for i in range(n_values):
            for j in range(n_values):
                delta[i, j] = (values[i] - values[j]) ** 2

    # Calculate observed disagreement D_o
    D_o = np.sum(coincidence_matrix * delta) / n_c

    # Calculate expected disagreement D_e
    n_k = np.sum(coincidence_matrix, axis=1)  # Marginal totals

    D_e = 0.0
    for i in range(n_values):
        for j in range(n_values):
            if i != j:
                D_e += n_k[i] * n_k[j] * delta[i, j]

    D_e = D_e / (n_c * (n_c - 1))

    # Calculate alpha
    if D_e == 0:
        return 1.0 if D_o == 0 else 0.0

    alpha = 1 - (D_o / D_e)

    return alpha


# ============================================================================
# FLEISS' KAPPA
# ============================================================================

def fleiss_kappa(data_matrix: np.ndarray, categories: List = None) -> float:
    """
    Calculate Fleiss' Kappa for multi-rater agreement

    Args:
        data_matrix: Matrix of shape (n_raters, n_items)
        categories: Optional list of categories (auto-detected if None)

    Returns:
        Fleiss' Kappa value (-1 to 1)
    """

    # Check for missing data - convert NaN to a special category
    if np.any(np.isnan(data_matrix)):
        # Add -999 as "missing" category
        data_matrix = np.where(np.isnan(data_matrix), -999, data_matrix)

    n_raters, n_items = data_matrix.shape

    # Get categories
    if categories is None:
        categories = np.unique(data_matrix)

    n_categories = len(categories)

    # Create category to index mapping
    cat_to_idx = {cat: i for i, cat in enumerate(categories)}

    # Build matrix of counts: rows = items, cols = categories
    count_matrix = np.zeros((n_items, n_categories))

    for item_idx in range(n_items):
        for rater_idx in range(n_raters):
            value = data_matrix[rater_idx, item_idx]
            if value in cat_to_idx:
                cat_idx = cat_to_idx[value]
                count_matrix[item_idx, cat_idx] += 1

    # Calculate P_i: extent of agreement for each item
    P_i = (np.sum(count_matrix ** 2, axis=1) - n_raters) / (n_raters * (n_raters - 1))

    # Calculate P_bar: average extent of agreement
    P_bar = np.mean(P_i)

    # Calculate P_e_bar: expected agreement by chance
    p_j = np.sum(count_matrix, axis=0) / (n_items * n_raters)
    P_e_bar = np.sum(p_j ** 2)

    # Calculate Kappa
    if P_e_bar == 1.0:
        kappa = 1.0 if P_bar == 1.0 else 0.0
    else:
        kappa = (P_bar - P_e_bar) / (1 - P_e_bar)

    return kappa


# ============================================================================
# DATA LOADING
# ============================================================================

def load_annotations(filepath: str,
                    target_models: List[str] = None,
                    target_sizes: List[str] = None) -> List[Dict]:
    """
    Load annotations from JSONL file with optional filtering

    Args:
        filepath: Path to JSONL file
        target_models: List of model names to include (e.g., ['Qwen3-0.6B', 'Qwen3-4B'])
        target_sizes: List of sizes to include (e.g., ['2', '3', '4'])

    Returns:
        List of filtered annotation dictionaries
    """
    annotations = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)

                # Only keep annotated samples
                if not data.get('human_annotation'):
                    continue

                # Filter by model
                if target_models is not None:
                    model_name = data.get('model_name', '')
                    # Check if any target model is in the model_name
                    if not any(target in model_name for target in target_models):
                        continue

                # Filter by size
                if target_sizes is not None:
                    size = data.get('size', '')
                    if size not in target_sizes:
                        continue

                annotations.append(data)

    return annotations


def save_common_samples(annotator_data_list: List[List[Dict]],
                       annotator_names: List[str],
                       output_dir: str = './'):
    """
    Save common samples across all annotators as JSONL files

    Creates:
    - {annotator_name}_common.jsonl for each annotator (filtered common samples)
    - common_samples_summary.txt: Summary statistics
    - prediction_breakdown.txt: Shows which instances have multiple predictions
    """

    # Find common instances
    all_instance_ids = []
    for data in annotator_data_list:
        instance_ids = set(ann['instance_id'] for ann in data)
        all_instance_ids.append(instance_ids)

    common_instances = set.intersection(*all_instance_ids)
    common_instances = sorted(common_instances)

    # Save each annotator's common samples as JSONL
    for ann_name, data in zip(annotator_names, annotator_data_list):
        output_file = f"{output_dir}/{ann_name}_common.jsonl"

        # Filter to common instances
        common_data = [ann for ann in data if ann['instance_id'] in common_instances]

        # Sort by instance_id for consistency
        common_data = sorted(common_data, key=lambda x: x['instance_id'])

        # Write to JSONL
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in common_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # Generate summary statistics
    all_judgments = []
    for data in annotator_data_list:
        judgments = extract_part1_judgments(data, merge_wrong=False)
        all_judgments.append(judgments)

    # Find common predictions
    common_pred_keys = set(all_judgments[0].keys())
    for judgments in all_judgments[1:]:
        common_pred_keys &= set(judgments.keys())

    # Analyze prediction breakdown
    pred_by_instance = {}
    for pred_key in common_pred_keys:
        instance_id = pred_key.split('_pred')[0]
        if instance_id not in pred_by_instance:
            pred_by_instance[instance_id] = []
        pred_by_instance[instance_id].append(pred_key)

    # Save prediction breakdown
    with open(f"{output_dir}/prediction_breakdown.txt", 'w', encoding='utf-8') as f:
        f.write("PREDICTION BREAKDOWN\n")
        f.write("=" * 80 + "\n\n")
        f.write("Shows why there are 90 predictions from 60 instances\n\n")

        # Count instances by number of predictions
        pred_count_dist = {}
        for instance_id, pred_keys in pred_by_instance.items():
            n_preds = len(pred_keys)
            pred_count_dist[n_preds] = pred_count_dist.get(n_preds, 0) + 1

        f.write("Distribution:\n")
        total_preds = 0
        for n_preds in sorted(pred_count_dist.keys()):
            n_instances = pred_count_dist[n_preds]
            total_preds += n_preds * n_instances
            f.write(f"  {n_instances} instances with {n_preds} prediction(s) each = {n_preds * n_instances} predictions\n")

        f.write(f"\nTotal: {total_preds} predictions from {len(pred_by_instance)} instances\n")
        f.write(f"Average: {total_preds / len(pred_by_instance):.2f} predictions per instance\n\n")

        f.write("=" * 80 + "\n\n")

        # Show instances with multiple predictions
        f.write("INSTANCES WITH MULTIPLE PREDICTIONS:\n")
        f.write("-" * 80 + "\n\n")

        multi_pred_instances = [(inst_id, preds) for inst_id, preds in pred_by_instance.items() if len(preds) > 1]
        multi_pred_instances.sort(key=lambda x: len(x[1]), reverse=True)

        if multi_pred_instances:
            for instance_id, pred_keys in multi_pred_instances:
                f.write(f"{instance_id}: {len(pred_keys)} predictions\n")
                for pred_key in pred_keys:
                    f.write(f"  - {pred_key}\n")
                f.write("\n")
        else:
            f.write("(None - all instances have exactly 1 prediction)\n")

    # Save summary
    with open(f"{output_dir}/common_samples_summary.txt", 'w', encoding='utf-8') as f:
        f.write("COMMON SAMPLES SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Common instances: {len(common_instances)}\n")
        f.write(f"Common predictions: {len(common_pred_keys)}\n\n")

        f.write("Files generated:\n")
        for ann_name in annotator_names:
            f.write(f"  - {ann_name}_common.jsonl\n")
        f.write(f"  - prediction_breakdown.txt (explains 90 vs 60)\n")

        f.write("\n")
        f.write("Per-annotator statistics:\n")
        for ann_name, data, instance_ids in zip(annotator_names, annotator_data_list, all_instance_ids):
            common_data = [ann for ann in data if ann['instance_id'] in common_instances]
            judgments = extract_part1_judgments(data, merge_wrong=False)

            f.write(f"\n{ann_name}:\n")
            f.write(f"  Total annotated samples: {len(data)}\n")
            f.write(f"  Common samples saved: {len(common_data)}\n")
            f.write(f"  Total predictions judged: {len(judgments)}\n")


# ============================================================================
# PART 1: MODEL PREDICTION JUDGMENT AGREEMENT (TWO VERSIONS)
# ============================================================================

def extract_part1_judgments(annotations: List[Dict], merge_wrong: bool = False) -> Dict[str, List]:
    """
    Extract Part 1 judgments for each model prediction

    Args:
        annotations: List of annotation dictionaries
        merge_wrong: If True, merge 'wrong_type' and 'wrong_triple' into 'wrong'

    Returns:
        {
            'pred_key': [annotator1_judgment, annotator2_judgment, ...],
            ...
        }

    Judgment labels (strings):
        If merge_wrong=False (4 categories):
            'over_prediction' = Model wrong, actually correct (is_over_prediction=True)
            'correct' = Model prediction is right (is_correct=True)
            'wrong_type' = Model got error type wrong (wrong_reason='wrong_type')
            'wrong_triple' = Model got triple wrong (wrong_reason='wrong_triple')

        If merge_wrong=True (3 categories):
            'over_prediction' = Model wrong, actually correct
            'correct' = Model prediction is right
            'wrong' = Model prediction is wrong (merged wrong_type and wrong_triple)
    """

    judgments = defaultdict(list)

    for ann in annotations:
        instance_id = ann['instance_id']
        human_ann = ann.get('human_annotation', {})

        model_verif = human_ann.get('model_verification', [])

        # Handle "all correct confirmed" case
        if isinstance(model_verif, list) and len(model_verif) == 1:
            if model_verif[0].get('all_correct_confirmed'):
                # Model predicted no errors and human confirmed
                # No predictions to judge → skip
                continue

        for idx, pred in enumerate(model_verif):
            # Create unique key for this prediction
            pred_type = pred.get('predicted_type', '')
            pred_triple = pred.get('predicted_triple', {})

            # Use combination of instance + prediction index as key
            pred_key = f"{instance_id}_pred{idx}"

            # Encode judgment as string label
            if pred.get('is_over_prediction'):
                judgment = 'over_prediction'
            elif pred.get('is_correct'):
                judgment = 'correct'
            else:
                wrong_reason = pred.get('wrong_reason')
                if merge_wrong:
                    # Merge both wrong types into 'wrong'
                    if wrong_reason in ['wrong_type', 'wrong_triple']:
                        judgment = 'wrong'
                    else:
                        judgment = None  # Missing judgment
                else:
                    # Keep original 4 categories
                    if wrong_reason == 'wrong_type':
                        judgment = 'wrong_type'
                    elif wrong_reason == 'wrong_triple':
                        judgment = 'wrong_triple'
                    else:
                        judgment = None  # Missing judgment

            judgments[pred_key].append(judgment)

    return judgments


def calculate_part1_agreement(annotator_data_list: List[List[Dict]],
                              merge_wrong: bool = False) -> Dict:
    """
    Calculate agreement on Part 1 judgments

    Args:
        annotator_data_list: List of annotation lists from each annotator
        merge_wrong: If True, merge 'wrong_type' and 'wrong_triple' into 'wrong'

    Returns:
        {
            'krippendorff_alpha': float,
            'fleiss_kappa': float,
            'n_predictions': int,
            'judgment_distribution': dict,
            'version': str ('4-category' or '3-category')
        }
    """

    # Extract judgments from each annotator
    all_judgments = []
    for data in annotator_data_list:
        judgments = extract_part1_judgments(data, merge_wrong=merge_wrong)
        all_judgments.append(judgments)

    # Find common predictions across all annotators
    common_pred_keys = set(all_judgments[0].keys())
    for judgments in all_judgments[1:]:
        common_pred_keys &= set(judgments.keys())

    if not common_pred_keys:
        return {
            'krippendorff_alpha': np.nan,
            'fleiss_kappa': np.nan,
            'n_predictions': 0,
            'judgment_distribution': {},
            'version': '3-category' if merge_wrong else '4-category'
        }

    common_pred_keys = sorted(common_pred_keys)
    n_raters = len(annotator_data_list)
    n_items = len(common_pred_keys)

    # Define mapping from string labels to numeric codes for calculation
    if merge_wrong:
        label_to_code = {
            'over_prediction': 0,
            'correct': 1,
            'wrong': 2,
            None: np.nan
        }
    else:
        label_to_code = {
            'over_prediction': 0,
            'correct': 1,
            'wrong_type': 2,
            'wrong_triple': 3,
            None: np.nan
        }

    # Build data matrix: rows = raters, cols = predictions
    data_matrix = np.zeros((n_raters, n_items))

    for rater_idx, judgments in enumerate(all_judgments):
        for item_idx, pred_key in enumerate(common_pred_keys):
            judgment_list = judgments[pred_key]
            # Take first judgment (should only be one per annotator)
            label = judgment_list[0]
            data_matrix[rater_idx, item_idx] = label_to_code[label]

    # Calculate metrics
    alpha = krippendorff_alpha(data_matrix, level_of_measurement='nominal')
    kappa = fleiss_kappa(data_matrix)

    # Distribution of judgments (use string labels)
    distribution = {}
    for label, code in label_to_code.items():
        if label is not None:  # Skip None
            count = np.sum(data_matrix == code)
            distribution[label] = int(count)

    return {
        'krippendorff_alpha': alpha,
        'fleiss_kappa': kappa,
        'n_predictions': n_items,
        'judgment_distribution': distribution,
        'version': '3-category' if merge_wrong else '4-category'
    }


# ============================================================================
# PAIRWISE AGREEMENT CALCULATION (WITH TWO VERSIONS)
# ============================================================================

def calculate_pairwise_agreements(annotator_data_list: List[List[Dict]],
                                  annotator_names: List[str]) -> Dict:
    """
    Calculate pairwise agreement between each pair of annotators
    For both 4-category and 3-category versions of Part 1

    Returns:
        {
            'pair_name': {
                'part1_4cat': {'alpha': float, 'kappa': float},
                'part1_3cat': {'alpha': float, 'kappa': float},
                'part2_alpha': {...},
                'part2_kappa': {...},
                'n_predictions': int,
                'n_instances': int
            }
        }
    """

    results = {}

    n_annotators = len(annotator_names)

    for i in range(n_annotators):
        for j in range(i + 1, n_annotators):
            name1 = annotator_names[i]
            name2 = annotator_names[j]
            pair_name = f"{name1}_vs_{name2}"

            # Calculate Part 1 agreement (4-category version)
            part1_4cat = calculate_part1_agreement([annotator_data_list[i], annotator_data_list[j]],
                                                   merge_wrong=False)

            # Calculate Part 1 agreement (3-category version)
            part1_3cat = calculate_part1_agreement([annotator_data_list[i], annotator_data_list[j]],
                                                   merge_wrong=True)

            # Calculate Part 2 agreement
            part2_result = calculate_part2_agreement([annotator_data_list[i], annotator_data_list[j]])

            results[pair_name] = {
                'part1_4cat': {
                    'alpha': part1_4cat['krippendorff_alpha'],
                    'kappa': part1_4cat['fleiss_kappa']
                },
                'part1_3cat': {
                    'alpha': part1_3cat['krippendorff_alpha'],
                    'kappa': part1_3cat['fleiss_kappa']
                },
                'part2_alpha': part2_result['krippendorff_alpha'],
                'part2_kappa': part2_result['fleiss_kappa'],
                'n_predictions': part1_4cat['n_predictions'],
                'n_instances': part2_result['n_instances']
            }

    return results


# ============================================================================
# PART 2: SUPPLEMENTED ERROR COUNT AGREEMENT
# ============================================================================

def extract_part2_counts(annotations: List[Dict]) -> Dict[str, Dict[str, int]]:
    """
    Extract Part 2 error counts for each instance

    Returns:
        {
            'instance_id': {
                'missing': count,
                'extra': count,
                'incorrect': count,
                'total': count
            },
            ...
        }
    """

    counts = {}

    for ann in annotations:
        instance_id = ann['instance_id']
        human_ann = ann.get('human_annotation', {})

        missed = human_ann.get('missed_errors', {})

        missing_count = len(missed.get('missing', []))
        extra_count = len(missed.get('extra', []))
        incorrect_count = len(missed.get('incorrect', []))

        counts[instance_id] = {
            'missing': missing_count,
            'extra': extra_count,
            'incorrect': incorrect_count,
            'total': missing_count + extra_count + incorrect_count
        }

    return counts


def calculate_part2_agreement(annotator_data_list: List[List[Dict]]) -> Dict:
    """
    Calculate agreement on Part 2 supplemented error counts

    Returns:
        {
            'krippendorff_alpha': {'missing': ..., 'extra': ..., 'incorrect': ..., 'total': ...},
            'fleiss_kappa': {...},
            'n_instances': int
        }
    """

    # Extract counts from each annotator
    all_counts = []
    for data in annotator_data_list:
        counts = extract_part2_counts(data)
        all_counts.append(counts)

    # Find common instances
    common_ids = set(all_counts[0].keys())
    for counts in all_counts[1:]:
        common_ids &= set(counts.keys())

    if not common_ids:
        return {
            'krippendorff_alpha': {},
            'fleiss_kappa': {},
            'n_instances': 0
        }

    common_ids = sorted(common_ids)
    n_raters = len(annotator_data_list)
    n_items = len(common_ids)

    results = {
        'krippendorff_alpha': {},
        'fleiss_kappa': {},
        'n_instances': n_items
    }

    # Calculate for each error type
    for error_type in ['missing', 'extra', 'incorrect', 'total']:
        # Build data matrix
        data_matrix = np.zeros((n_raters, n_items))

        for rater_idx, counts in enumerate(all_counts):
            for item_idx, instance_id in enumerate(common_ids):
                count = counts[instance_id][error_type]
                data_matrix[rater_idx, item_idx] = count

        # Calculate metrics
        alpha = krippendorff_alpha(data_matrix, level_of_measurement='interval')
        kappa = fleiss_kappa(data_matrix)

        results['krippendorff_alpha'][error_type] = alpha
        results['fleiss_kappa'][error_type] = kappa

    return results


# ============================================================================
# MAIN REPORT GENERATION
# ============================================================================

def generate_report(annotator_data_list: List[List[Dict]],
                   annotator_names: List[str],
                   output_path: str,
                   target_models: List[str] = None,
                   target_sizes: List[str] = None):
    """Generate comprehensive agreement report with both Part 1 versions"""

    lines = []

    lines.append("=" * 80)
    lines.append("ANNOTATION PROCESS AGREEMENT ANALYSIS")
    lines.append("=" * 80)
    lines.append("")

    lines.append(f"Number of annotators: {len(annotator_names)}")
    lines.append(f"Annotators: {', '.join(annotator_names)}")

    # Add filtering information
    if target_models or target_sizes:
        lines.append("")
        lines.append("FILTERING CRITERIA:")
        if target_models:
            lines.append(f"  Target models: {', '.join(target_models)}")
        if target_sizes:
            lines.append(f"  Target sizes: {', '.join(target_sizes)}")

    lines.append("")

    # ========================================================================
    # PART 1A: MODEL PREDICTION JUDGMENT AGREEMENT (4-CATEGORY VERSION)
    # ========================================================================

    lines.append("=" * 80)
    lines.append("PART 1A: AGREEMENT ON MODEL PREDICTIONS (4-CATEGORY VERSION)")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Categories: over_prediction, correct, wrong_type, wrong_triple")
    lines.append("")

    part1_4cat = calculate_part1_agreement(annotator_data_list, merge_wrong=False)

    lines.append(f"Number of common predictions evaluated: {part1_4cat['n_predictions']}")
    lines.append("")

    lines.append("Judgment Categories:")
    lines.append("  'over_prediction' = Model wrong, actually correct")
    lines.append("  'correct'         = Model prediction is right")
    lines.append("  'wrong_type'      = Model got error type wrong")
    lines.append("  'wrong_triple'    = Model got triple wrong")
    lines.append("")

    lines.append("Judgment Distribution:")
    for category, count in part1_4cat['judgment_distribution'].items():
        lines.append(f"  {category:20s}: {count:4d}")
    lines.append("")

    lines.append("METRICS:")
    lines.append("-" * 80)

    alpha = part1_4cat['krippendorff_alpha']
    kappa = part1_4cat['fleiss_kappa']

    lines.append(f"Krippendorff's Alpha: {alpha:.4f}" if not np.isnan(alpha) else "Krippendorff's Alpha: NaN")
    lines.append(f"Fleiss' Kappa:        {kappa:.4f}" if not np.isnan(kappa) else "Fleiss' Kappa:        NaN")
    lines.append("")

    # ========================================================================
    # PART 1B: MODEL PREDICTION JUDGMENT AGREEMENT (3-CATEGORY VERSION)
    # ========================================================================

    lines.append("=" * 80)
    lines.append("PART 1B: AGREEMENT ON MODEL PREDICTIONS (3-CATEGORY VERSION)")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Categories: over_prediction, correct, wrong (merged wrong_type + wrong_triple)")
    lines.append("")

    part1_3cat = calculate_part1_agreement(annotator_data_list, merge_wrong=True)

    lines.append(f"Number of common predictions evaluated: {part1_3cat['n_predictions']}")
    lines.append("")

    lines.append("Judgment Categories:")
    lines.append("  'over_prediction' = Model wrong, actually correct")
    lines.append("  'correct'         = Model prediction is right")
    lines.append("  'wrong'           = Model prediction is wrong (any type)")
    lines.append("")

    lines.append("Judgment Distribution:")
    for category, count in part1_3cat['judgment_distribution'].items():
        lines.append(f"  {category:20s}: {count:4d}")
    lines.append("")

    lines.append("METRICS:")
    lines.append("-" * 80)

    alpha = part1_3cat['krippendorff_alpha']
    kappa = part1_3cat['fleiss_kappa']

    lines.append(f"Krippendorff's Alpha: {alpha:.4f}" if not np.isnan(alpha) else "Krippendorff's Alpha: NaN")
    lines.append(f"Fleiss' Kappa:        {kappa:.4f}" if not np.isnan(kappa) else "Fleiss' Kappa:        NaN")
    lines.append("")

    # ========================================================================
    # PART 2: SUPPLEMENTED ERROR COUNT AGREEMENT
    # ========================================================================

    lines.append("=" * 80)
    lines.append("PART 2: AGREEMENT ON SUPPLEMENTED ERROR COUNTS")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Measures: How consistently do annotators supplement missed errors?")
    lines.append("")

    part2_results = calculate_part2_agreement(annotator_data_list)

    lines.append(f"Number of common instances: {part2_results['n_instances']}")
    lines.append("")

    lines.append("KRIPPENDORFF'S ALPHA (interval level):")
    lines.append("-" * 80)
    for error_type, alpha in part2_results['krippendorff_alpha'].items():
        if not np.isnan(alpha):
            lines.append(f"  {error_type:15s}: {alpha:.4f}")
        else:
            lines.append(f"  {error_type:15s}: NaN")
    lines.append("")

    # ========================================================================
    # PAIRWISE AGREEMENT (All pairs, both Part 1 versions)
    # ========================================================================

    lines.append("=" * 80)
    lines.append("PAIRWISE AGREEMENT (Between Each Pair of Annotators)")
    lines.append("=" * 80)
    lines.append("")

    pairwise_results = calculate_pairwise_agreements(annotator_data_list, annotator_names)

    for pair_name, metrics in pairwise_results.items():
        lines.append(f"{pair_name}:")
        lines.append("-" * 80)
        lines.append("")

        lines.append("Part 1A (4-Category: over_pred, correct, wrong_type, wrong_triple):")
        lines.append(f"  Common predictions: {metrics['n_predictions']}")

        alpha = metrics['part1_4cat']['alpha']
        kappa = metrics['part1_4cat']['kappa']
        if not np.isnan(alpha):
            lines.append(f"  Krippendorff's Alpha: {alpha:.4f}")
        else:
            lines.append(f"  Krippendorff's Alpha: NaN")

        if not np.isnan(kappa):
            lines.append(f"  Fleiss' Kappa:        {kappa:.4f}")
        else:
            lines.append(f"  Fleiss' Kappa:        NaN")

        lines.append("")

        lines.append("Part 1B (3-Category: over_pred, correct, wrong):")
        alpha = metrics['part1_3cat']['alpha']
        kappa = metrics['part1_3cat']['kappa']
        if not np.isnan(alpha):
            lines.append(f"  Krippendorff's Alpha: {alpha:.4f}")
        else:
            lines.append(f"  Krippendorff's Alpha: NaN")

        if not np.isnan(kappa):
            lines.append(f"  Fleiss' Kappa:        {kappa:.4f}")
        else:
            lines.append(f"  Fleiss' Kappa:        NaN")

        lines.append("")

        lines.append("Part 2 (Supplemented Error Counts):")
        lines.append(f"  Common instances: {metrics['n_instances']}")

        lines.append("  Krippendorff's Alpha:")
        for error_type, alpha in metrics['part2_alpha'].items():
            if not np.isnan(alpha):
                lines.append(f"    {error_type:15s}: {alpha:.4f}")
            else:
                lines.append(f"    {error_type:15s}: NaN")

        lines.append("")
        lines.append("")

    # ========================================================================
    # INTERPRETATION
    # ========================================================================

    lines.append("=" * 80)
    lines.append("INTERPRETATION GUIDELINES")
    lines.append("=" * 80)
    lines.append("")

    lines.append("Krippendorff's Alpha:")
    lines.append("  α ≥ 0.800: Excellent agreement")
    lines.append("  0.667 ≤ α < 0.800: Good agreement (tentative conclusions)")
    lines.append("  0.500 ≤ α < 0.667: Moderate agreement (use with caution)")
    lines.append("  α < 0.500: Poor agreement (not reliable)")
    lines.append("")

    lines.append("Fleiss' Kappa:")
    lines.append("  κ ≥ 0.80: Almost perfect agreement")
    lines.append("  0.60 ≤ κ < 0.80: Substantial agreement")
    lines.append("  0.40 ≤ κ < 0.60: Moderate agreement")
    lines.append("  0.20 ≤ κ < 0.40: Fair agreement")
    lines.append("  κ < 0.20: Slight agreement")
    lines.append("")

    # Save report
    report = "\n".join(lines)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution"""

    # File paths
    input_files = [
        BASE / 'human_annotations_agreement_1.jsonl',
        BASE / 'human_annotations_agreement_2.jsonl',
        BASE / 'human_annotations_agreement_3.jsonl'
    ]

    annotator_names = ['annotator_1', 'annotator_2', 'annotator_3']

    # Filtering criteria
    target_models = ['Qwen3-0.6B', 'Qwen3-4B']  # Only Qwen3 0.6B and 4B
    target_sizes = ['2', '3', '4']              # Only size 2, 3, 4

    # Load annotations with filtering
    all_annotator_data = []
    for filepath in input_files:
        annotations = load_annotations(filepath,
                                      target_models=target_models,
                                      target_sizes=target_sizes)
        all_annotator_data.append(annotations)

    if len(all_annotator_data) < 2:
        return

    # Save common samples for inspection
    save_common_samples(all_annotator_data, annotator_names, output_dir=BASE)

    # Generate report
    report_path = BASE / "annotation_process_agreement.txt"
    generate_report(all_annotator_data, annotator_names, report_path,
                   target_models=target_models, target_sizes=target_sizes)


if __name__ == "__main__":
    main()
