"""
KELM Annotation Analysis - Final
Model performance comparison: P/R/F1, error pattern analysis,
size bucketing, error density bucketing.

Data source (3 de-identified annotation files):
  - human_annotations_main_1.jsonl : 0.6B x60 + 4B x60
  - human_annotations_main_2.jsonl : 8B x60 + 0.6B x30 + 4B x30
  - human_annotations_main_3.jsonl : 14B x60 + 0.6B x30 + 4B x30

Ground truth construction:
  - 8B  (60 samples) : annotator 2 only
  - 14B (60 samples) : annotator 3 only
  - 0.6B / 4B non-overlap (30+30) : annotator 1 only
  - 0.6B / 4B overlap (30+30)     : all 3 annotators merged (mean for numerics, majority for booleans)
"""

import json
import sys
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path

BASE = Path(__file__).parent


class _Tee:
    """Write to both stdout and a log file simultaneously."""
    def __init__(self, log_path):
        self._stdout = sys.stdout
        self._file = open(log_path, 'w', encoding='utf-8')

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        self._file.close()
        sys.stdout = self._stdout


def load_jsonl(path):
    data = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def merge_multi_annotator(entries: list) -> dict:
    """
    Merge annotations from multiple annotators for the same instance.

    All human_annotation objects are preserved in 'all_annotations' so that
    compute_gold_errors() and extract_error_pattern() can average over all of them.

    P/R/F1 are averaged across annotators (mean), since each annotator's
    scores are computed against their own ground truth judgments.
    """
    assert len(entries) >= 2
    base = entries[0]

    return {
        **base,
        'n_annotators':    len(entries),
        'all_annotations': [e['human_annotation'] for e in entries],
        'human_annotation': None,
    }


def build_ground_truth():
    """
    Build the final 240-sample ground truth.

    Overlap detection:
      - Load 0.6B/4B entries from all 3 files.
      - instance_ids appearing in all 3 files -> merged (3 annotators).
      - instance_ids appearing only in main_1 -> one annotation.
      - 8B rows from main_2 and 14B rows from main_3 -> one annotation.
    """
    file_1 = BASE / 'human_annotations_main_1.jsonl'
    file_2 = BASE / 'human_annotations_main_2.jsonl'
    file_3 = BASE / 'human_annotations_main_3.jsonl'

    def is_06_or_4b(d):
        name = d['model_name']
        return 'Qwen3-0.6B' in name or 'Qwen3-4B' in name

    annotator_1 = {d['instance_id']: d for d in load_jsonl(file_1)
               if d['human_annotation'] and is_06_or_4b(d)}
    annotator_2 = {d['instance_id']: d for d in load_jsonl(file_2)
               if d['human_annotation'] and is_06_or_4b(d)}
    annotator_3 = {d['instance_id']: d for d in load_jsonl(file_3)
               if d['human_annotation'] and is_06_or_4b(d)}

    overlap_ids = set(annotator_1) & set(annotator_2) & set(annotator_3)
    annotator_1_only_ids = set(annotator_1) - overlap_ids

    ground_truth = []

    for iid in overlap_ids:
        entries = [annotator_1[iid], annotator_2[iid], annotator_3[iid]]
        ground_truth.append(merge_multi_annotator(entries))

    for iid in annotator_1_only_ids:
        ground_truth.append({**annotator_1[iid], 'n_annotators': 1})

    for d in load_jsonl(file_2):
        if d['human_annotation'] and 'Qwen3-8B' in d['model_name']:
            ground_truth.append({**d, 'n_annotators': 1})

    for d in load_jsonl(file_3):
        if d['human_annotation'] and 'Qwen3-14B' in d['model_name']:
            ground_truth.append({**d, 'n_annotators': 1})

    print(f"Ground truth built: {len(ground_truth)} samples")
    model_counts = Counter(d['model_name'] for d in ground_truth)
    for m, n in sorted(model_counts.items()):
        n3 = sum(1 for d in ground_truth
                 if d['model_name'] == m and d['n_annotators'] == 3)
        print(f"  {m}: {n} samples  (3-annotator merged: {n3})")

    return ground_truth


def _gold_errors_from_ann(ann: dict) -> dict:
    """
    Compute gold error counts from a single human_annotation dict.

    Gold = everything the human confirmed as a real error:
      Part1 correct     : model predicted right -> gold type = predicted_type
      Part1 wrong_type  : triple right, type wrong -> gold type = correct_type
      Part1 wrong_triple: type right, triple wrong -> gold type = predicted_type
      Part1 over_pred   : human says no real error -> NOT gold
      Part2 missed      : errors model completely missed, human supplemented
    """
    mv = ann.get('model_verification', [])

    p1_missing = p1_extra = p1_incorrect = 0
    for p in mv:
        if p.get('all_correct_confirmed') or p.get('is_over_prediction'):
            continue
        wr = p.get('wrong_reason', '')
        if wr == 'wrong_type':
            # triple is correct, but the error type was mislabelled -> use correct_type
            gold_type = p.get('correct_type', '')
        else:
            # is_correct or wrong_triple: the predicted type stands
            gold_type = p.get('predicted_type', '')
        if gold_type == 'missing':
            p1_missing += 1
        elif gold_type == 'extra':
            p1_extra += 1
        elif gold_type == 'incorrect':
            p1_incorrect += 1

    missed = ann.get('missed_errors', {})
    p2_missing   = len(missed.get('missing', []))
    p2_extra     = len(missed.get('extra', []))
    p2_incorrect = len(missed.get('incorrect', []))

    return {
        'missing':   p1_missing + p2_missing,
        'extra':     p1_extra   + p2_extra,
        'incorrect': p1_incorrect + p2_incorrect,
        'total':     p1_missing + p1_extra + p1_incorrect + p2_missing + p2_extra + p2_incorrect,
    }


def _prf_from_ann(ann: dict, strict: bool = False, acc_tp: bool = False,
                  exclude_acc: bool = False):
    """
    Recompute precision / recall / F1 from human annotation.

    Two evaluation modes for wrong_type / wrong_triple predictions:
      strict=False (lenient): model found the right error triple, just mislabelled it.
        -> only FN (partial credit: the error was detected, type was wrong)
        -> analogous to nervaluate "Exact" mode

      strict=True (CoNLL-style): wrong label = predicted something that does not exist
        -> FP + FN (double penalty, same as conlleval / seqeval default)
        -> analogous to nervaluate "Strict" mode

    In both modes:
      TP = Part1 is_correct
      FP = Part1 is_over_prediction  (+ wrong_type/wrong_triple in strict mode)
      FN = Part1 wrong_type + wrong_triple  (+ missed errors)

    acc_tp: when model predicted no errors and human confirms all-correct,
      True  -> treat as TP=1 (model made a correct all-correct judgment)
      False -> treat as TP=0, P=1.0, R=1.0 by convention (no predictions made)

    exclude_acc: when model predicted no errors AND human confirms all-correct
      AND there are no missed errors (a "perfect all-correct" sample),
      return None so the caller can skip this sample from macro averaging.
      Samples where all-correct was confirmed but there ARE missed errors are
      kept (the model silently missed real errors -> meaningful signal).
    """
    mv = ann.get('model_verification', [])

    # All entries are all_correct_confirmed (covers mv=[], mv=[{acc}], and mv=[{acc},{acc},...])
    # FP = 0; FN = missed errors only
    if len(mv) == 0 or all(p.get('all_correct_confirmed') for p in mv):
        missed = ann.get('missed_errors', {})
        fn = (len(missed.get('missing', [])) +
              len(missed.get('extra', [])) +
              len(missed.get('incorrect', [])))
        # exclude_acc: skip perfect all-correct samples (no missed errors) from macro
        if exclude_acc and fn == 0:
            return None
        if acc_tp:
            tp = 1
            precision = 1.0
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        else:
            tp = 0
            precision = 1.0 if fn == 0 else 0.0
            recall    = 1.0 if fn == 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        return {'precision': precision, 'recall': recall, 'f1': f1,
                'TP': tp, 'FP': 0, 'FN': fn}

    tp = fp = fn = 0
    for pred in mv:
        if pred.get('all_correct_confirmed'):
            continue
        if pred.get('is_over_prediction'):
            fp += 1
        elif pred.get('is_correct'):
            tp += 1
        else:
            # wrong_type or wrong_triple
            fn += 1
            if strict:
                fp += 1  # double penalty: the mislabelled prediction also counts as a false alarm

    missed = ann.get('missed_errors', {})
    fn += (len(missed.get('missing', [])) +
           len(missed.get('extra', [])) +
           len(missed.get('incorrect', [])))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return {'precision': precision, 'recall': recall, 'f1': f1,
            'TP': tp, 'FP': fp, 'FN': fn}


def _typewise_tpfpfn_from_ann(ann: dict) -> dict:
    """
    Compute per-error-type TP/FP/FN from a single human_annotation dict (lenient mode).

    For each error type t in {missing, extra, incorrect}:
      TP_t = model predicted type t AND human confirms the triple is a real error of type t
             (is_correct with predicted_type==t,
              OR wrong_type with correct_type==t -> triple right, type was t after correction... wait:
              wrong_type means predicted_type wrong -> NOT TP for predicted type)
      More carefully (lenient):
        - is_correct, predicted_type==t          -> TP for t
        - is_over_prediction                     -> FP for predicted_type (no TP)
        - wrong_type: triple right, type wrong
            predicted_type==t                   -> FP for t (predicted t, but gold type is correct_type)
            correct_type==t                     -> FN for t (gold is t, but model said other type -> missed)
        - wrong_triple: type right, triple wrong
            predicted_type==t                   -> FP for t (predicted t wrong triple)
            gold type is t (predicted_type==t)  -> also FN for t via Part2 missed? No:
            wrong_triple: type correct but triple wrong -> that real error is still in gold as FN
            So: wrong_triple, predicted_type==t -> FP_t (wrong triple) AND the actual error
            becomes a Part2-style miss -> FN_t counted via gold
      FN_t = gold errors of type t that model did NOT correctly predict
           = (gold_t) - TP_t
           where gold_t = _gold_errors_from_ann()[t]
      FP_t = predictions of type t that are NOT TP_t
           = predicted_as_t - TP_t

    Simplest correct formulation:
      For each prediction p:
        if is_correct and predicted_type==t  -> TP_t += 1
        if (is_over_prediction or wrong_triple or wrong_type) and predicted_type==t -> FP_t += 1
      gold_t = _gold_errors_from_ann()[t]
      FN_t = gold_t - TP_t   (gold errors of type t that weren't correctly predicted)
    """
    mv = ann.get('model_verification', [])

    tp = {'missing': 0, 'extra': 0, 'incorrect': 0}
    fp = {'missing': 0, 'extra': 0, 'incorrect': 0}

    if not (len(mv) == 0 or all(p.get('all_correct_confirmed') for p in mv)):
        for pred in mv:
            if pred.get('all_correct_confirmed'):
                continue
            pt = pred.get('predicted_type', '')
            if pt not in tp:
                continue
            if pred.get('is_correct'):
                tp[pt] += 1
            else:
                # is_over_prediction, wrong_type, or wrong_triple: all count as FP for predicted type
                fp[pt] += 1

    gold = _gold_errors_from_ann(ann)
    fn = {t: max(0, gold[t] - tp[t]) for t in ('missing', 'extra', 'incorrect')}

    return {'tp': tp, 'fp': fp, 'fn': fn}


def compute_typewise_tpfpfn(d: dict) -> dict:
    """
    Compute per-error-type TP/FP/FN for a sample.
    If multiple annotators, average TP/FP/FN across annotators.
    Returns dict with keys 'tp', 'fp', 'fn', each a sub-dict of {missing, extra, incorrect}.
    """
    all_anns = d.get('all_annotations')
    if all_anns:
        results = [_typewise_tpfpfn_from_ann(a) for a in all_anns if a]
        if not results:
            return {'tp': {'missing':0,'extra':0,'incorrect':0},
                    'fp': {'missing':0,'extra':0,'incorrect':0},
                    'fn': {'missing':0,'extra':0,'incorrect':0}}
        averaged = {}
        for key in ('tp', 'fp', 'fn'):
            averaged[key] = {t: float(np.mean([r[key][t] for r in results]))
                             for t in ('missing', 'extra', 'incorrect')}
        return averaged
    ann = d.get('human_annotation') or {}
    return _typewise_tpfpfn_from_ann(ann)


def compute_prf(d: dict, strict: bool = False, acc_tp: bool = False,
                exclude_acc: bool = False):
    """
    Compute P/R/F1 for a sample, averaging over annotators if multi-annotator.
    Also returns averaged TP/FP/FN for micro aggregation.

    If exclude_acc=True, returns None for samples that are perfect all-correct
    (model said all-correct AND no missed errors), so the caller can skip them
    from macro averaging.  If a multi-annotator sample has some annotators
    returning None and some returning values, the non-None ones are averaged;
    the sample is excluded only if ALL annotators returned None.
    """
    all_anns = d.get('all_annotations')
    if all_anns:
        results = [_prf_from_ann(a, strict=strict, acc_tp=acc_tp,
                                 exclude_acc=exclude_acc)
                   for a in all_anns if a]
        # filter out None (annotators for whom this is a perfect all-correct)
        valid = [r for r in results if r is not None]
        if not valid:
            # all annotators returned None -> exclude this sample
            return None
        return {k: float(np.mean([r[k] for r in valid]))
                for k in ['precision', 'recall', 'f1', 'TP', 'FP', 'FN']}
    ann = d.get('human_annotation') or {}
    return _prf_from_ann(ann, strict=strict, acc_tp=acc_tp, exclude_acc=exclude_acc)


def compute_gold_errors(d: dict) -> dict:
    """
    Compute gold error counts for a sample.
    If multiple annotators exist (all_annotations), average their counts.
    """
    all_anns = d.get('all_annotations')
    if all_anns:
        results = [_gold_errors_from_ann(a) for a in all_anns if a]
        if not results:
            return {'missing': 0, 'extra': 0, 'incorrect': 0, 'total': 0}
        return {k: float(np.mean([r[k] for r in results]))
                for k in ['missing', 'extra', 'incorrect', 'total']}

    ann = d.get('human_annotation') or {}
    if not ann:
        return {'missing': 0, 'extra': 0, 'incorrect': 0, 'total': 0}
    return _gold_errors_from_ann(ann)


def _error_pattern_from_ann(ann: dict) -> dict:
    """Extract error pattern metrics from a single human_annotation dict."""
    mv = ann.get('model_verification', [])
    missed = ann.get('missed_errors', {})
    n_missed = (len(missed.get('missing', [])) +
                len(missed.get('extra', [])) +
                len(missed.get('incorrect', [])))

    model_said_all_correct = (
        len(mv) == 0 or
        (len(mv) == 1 and mv[0].get('all_correct_confirmed'))
    )

    n_over_pred = n_correct = n_wrong_type = n_wrong_triple = 0
    real_preds = [p for p in mv if not p.get('all_correct_confirmed')]
    for p in real_preds:
        if p.get('is_over_prediction'):
            n_over_pred += 1
        elif p.get('is_correct'):
            n_correct += 1
        else:
            wr = p.get('wrong_reason', '')
            if wr == 'wrong_type':
                n_wrong_type += 1
            elif wr == 'wrong_triple':
                n_wrong_triple += 1

    return {
        'n_predicted':            len(real_preds),
        'n_over_pred':            n_over_pred,
        'n_correct_pred':         n_correct,
        'n_wrong_type':           n_wrong_type,
        'n_wrong_triple':         n_wrong_triple,
        'model_said_all_correct': model_said_all_correct,
        'has_missed':             n_missed > 0,
        'n_missed':               n_missed,
    }


def extract_error_pattern(d: dict) -> dict:
    """
    Extract error pattern metrics for a sample.
    If multiple annotators exist (all_annotations), average numeric metrics
    and use majority vote for booleans.
    """
    all_anns = d.get('all_annotations')
    if all_anns:
        results = [_error_pattern_from_ann(a) for a in all_anns if a]
        if not results:
            return _error_pattern_from_ann({})
        averaged = {k: float(np.mean([r[k] for r in results]))
                    for k in ['n_predicted', 'n_over_pred', 'n_correct_pred',
                               'n_wrong_type', 'n_wrong_triple', 'n_missed']}
        averaged['model_said_all_correct'] = (
            sum(bool(r['model_said_all_correct']) for r in results) >= len(results) / 2)
        averaged['has_missed'] = (
            sum(bool(r['has_missed']) for r in results) >= len(results) / 2)
        return averaged

    ann = d.get('human_annotation') or {}
    return _error_pattern_from_ann(ann)


MODEL_ORDER = ['Qwen3-0.6B-8864', 'Qwen3-4B-4432', 'Qwen3-8B-7479', 'Qwen3-14B-6648']
MODEL_LABELS = {
    'Qwen3-0.6B-8864': '0.6B',
    'Qwen3-4B-4432':   '4B',
    'Qwen3-8B-7479':   '8B',
    'Qwen3-14B-6648':  '14B',
}


def analyse_group(group: list, strict: bool = False, acc_tp: bool = False) -> dict:
    """Aggregate all metrics for a list of samples."""
    if not group:
        return None

    prf_list = [compute_prf(d, strict=strict, acc_tp=acc_tp) for d in group]

    # Macro: average per-sample P/R/F1 (each sample weighted equally)
    p  = np.mean([x['precision'] for x in prf_list])
    r  = np.mean([x['recall']    for x in prf_list])
    f1 = np.mean([x['f1']        for x in prf_list])

    # Macro (excl): exclude perfect all-correct samples (no missed errors)
    # Only average over samples where the model actually made predictions OR
    # the model said all-correct but missed real errors.
    prf_excl = [compute_prf(d, strict=strict, acc_tp=acc_tp, exclude_acc=True)
                for d in group]
    prf_excl_valid = [x for x in prf_excl if x is not None]
    if prf_excl_valid:
        p_excl  = float(np.mean([x['precision'] for x in prf_excl_valid]))
        r_excl  = float(np.mean([x['recall']    for x in prf_excl_valid]))
        f1_excl = float(np.mean([x['f1']        for x in prf_excl_valid]))
        n_excl  = len(prf_excl_valid)
    else:
        p_excl = r_excl = f1_excl = float('nan')
        n_excl = 0

    # Micro: pool TP/FP/FN across all samples, then compute P/R/F1 once
    tp_sum = sum(x['TP'] for x in prf_list)
    fp_sum = sum(x['FP'] for x in prf_list)
    fn_sum = sum(x['FN'] for x in prf_list)
    micro_p  = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0.0
    micro_r  = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0.0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)
                if (micro_p + micro_r) > 0 else 0.0)

    golds = [compute_gold_errors(d) for d in group]
    avg_gold = np.mean([g['total'] for g in golds])

    error_patterns = [extract_error_pattern(d) for d in group]
    avg_pred = np.mean([b['n_predicted'] for b in error_patterns])

    n_preds_total = sum(b['n_predicted']    for b in error_patterns)
    n_over_total  = sum(b['n_over_pred']    for b in error_patterns)
    n_wtype_total = sum(b['n_wrong_type']   for b in error_patterns)
    n_wtrip_total = sum(b['n_wrong_triple'] for b in error_patterns)
    n_missed_total= sum(b['n_missed']       for b in error_patterns)
    # all mistakes: over-prediction + wrong type + wrong triple + missed errors
    n_mistakes_total = n_over_total + n_wtype_total + n_wtrip_total + n_missed_total

    # denom = total predictions (including correct ones)
    over_rate  = n_over_total  / n_preds_total if n_preds_total > 0 else np.nan
    wtype_rate = n_wtype_total / n_preds_total if n_preds_total > 0 else np.nan
    wtrip_rate = n_wtrip_total / n_preds_total if n_preds_total > 0 else np.nan

    # denom = total mistakes (over-pred + wrong type + wrong triple + missed)
    over_rate_m  = n_over_total  / n_mistakes_total if n_mistakes_total > 0 else np.nan
    wtype_rate_m = n_wtype_total / n_mistakes_total if n_mistakes_total > 0 else np.nan
    wtrip_rate_m = n_wtrip_total / n_mistakes_total if n_mistakes_total > 0 else np.nan
    miss_rate_m  = n_missed_total/ n_mistakes_total if n_mistakes_total > 0 else np.nan

    miss_rate = sum(b['has_missed'] for b in error_patterns) / len(error_patterns)

    # Binary classification: "all correct" = positive class, "has errors" = negative class
    #   model_said_all_correct=True  AND gold_errors == 0 -> binary TP (correctly said all-correct)
    #   model_said_all_correct=True  AND gold_errors > 0  -> binary FP (missed errors, said all-correct wrongly)
    #   model_said_all_correct=False AND gold_errors == 0 -> binary FN (said has-errors, but gold is all-correct)
    #   model_said_all_correct=False AND gold_errors > 0  -> binary TN (correctly flagged errors)
    bin_tp = sum(1 for d, b in zip(group, error_patterns)
                 if b['model_said_all_correct']     and d['gold_errors']['total'] < 0.5)
    bin_fp = sum(1 for d, b in zip(group, error_patterns)
                 if b['model_said_all_correct']     and d['gold_errors']['total'] >= 0.5)
    bin_fn = sum(1 for d, b in zip(group, error_patterns)
                 if not b['model_said_all_correct'] and d['gold_errors']['total'] < 0.5)
    bin_tn = sum(1 for d, b in zip(group, error_patterns)
                 if not b['model_said_all_correct'] and d['gold_errors']['total'] >= 0.5)

    bin_precision = bin_tp / (bin_tp + bin_fp) if (bin_tp + bin_fp) > 0 else float('nan')
    bin_recall    = bin_tp / (bin_tp + bin_fn) if (bin_tp + bin_fn) > 0 else float('nan')
    if not (np.isnan(bin_precision) or np.isnan(bin_recall)) and (bin_precision + bin_recall) > 0:
        bin_f1 = 2 * bin_precision * bin_recall / (bin_precision + bin_recall)
    else:
        bin_f1 = float('nan')
    bin_accuracy = (bin_tp + bin_tn) / len(group) if group else float('nan')

    return {
        'n':               len(group),
        'precision':       p,
        'recall':          r,
        'f1':              f1,
        'precision_excl':  p_excl,
        'recall_excl':     r_excl,
        'f1_excl':         f1_excl,
        'n_excl':          n_excl,
        'micro_precision': micro_p,
        'micro_recall':    micro_r,
        'micro_f1':        micro_f1,
        'avg_gold_errors': avg_gold,
        'avg_pred_errors': avg_pred,
        'pred_bias':       avg_pred - avg_gold,
        'over_pred_rate':  over_rate,
        'wrong_type_rate': wtype_rate,
        'wrong_trip_rate': wtrip_rate,
        'miss_rate':       miss_rate,
        'over_rate_m':     over_rate_m,
        'wtype_rate_m':    wtype_rate_m,
        'wtrip_rate_m':    wtrip_rate_m,
        'miss_rate_m':     miss_rate_m,
        'avg_missed':      np.mean([b['n_missed'] for b in error_patterns]),
        'bin_tp':          bin_tp,
        'bin_fp':          bin_fp,
        'bin_fn':          bin_fn,
        'bin_tn':          bin_tn,
        'bin_precision':   bin_precision,
        'bin_recall':      bin_recall,
        'bin_f1':          bin_f1,
        'bin_accuracy':    bin_accuracy,
    }


def _sample_prf(d: dict, metric: str, rng, strict: bool = False, acc_tp: bool = False,
                exclude_acc: bool = False):
    """
    For a single sample, return the metric value (or None if exclude_acc applies).
    For multi-annotator samples, randomly pick one annotator's annotation
    so bootstrap also captures annotator uncertainty.
    """
    all_anns = d.get('all_annotations')
    if all_anns:
        ann = all_anns[rng.integers(0, len(all_anns))]
        r = _prf_from_ann(ann, strict=strict, acc_tp=acc_tp, exclude_acc=exclude_acc)
        return r[metric] if r is not None else None
    ann = d.get('human_annotation') or {}
    r = _prf_from_ann(ann, strict=strict, acc_tp=acc_tp, exclude_acc=exclude_acc)
    return r[metric] if r is not None else None


def bootstrap_ci(group: list, metric: str, n_boot: int = 1000, alpha: float = 0.05,
                 seed: int = 2023, strict: bool = False) -> tuple:
    """
    Macro bootstrap CI: resample samples, compute mean of per-sample metric.
    For multi-annotator samples, randomly pick one annotator per draw.
    Returns (mean, lower, upper) at the given alpha level (default 95% CI).
    """
    rng = np.random.default_rng(seed)
    n = len(group)
    boot_means = []
    for _ in range(n_boot):
        indices = rng.integers(0, n, size=n)
        mean_val = np.mean([_sample_prf(group[i], metric, rng, strict=strict) for i in indices])
        boot_means.append(mean_val)
    boot_means = np.array(boot_means)
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(np.mean(boot_means)), lo, hi


def _sample_tpfpfn(d: dict, rng, strict: bool = False, acc_tp: bool = False) -> tuple:
    """
    For a single sample, return (TP, FP, FN).
    For multi-annotator samples, randomly pick one annotator per draw.
    """
    all_anns = d.get('all_annotations')
    if all_anns:
        ann = all_anns[rng.integers(0, len(all_anns))]
        r = _prf_from_ann(ann, strict=strict, acc_tp=acc_tp)
    else:
        ann = d.get('human_annotation') or {}
        r = _prf_from_ann(ann, strict=strict, acc_tp=acc_tp)
    return r['TP'], r['FP'], r['FN']


def bootstrap_micro_ci(group: list, n_boot: int = 1000, alpha: float = 0.05,
                       seed: int = 2023, strict: bool = False) -> tuple:
    """
    Micro bootstrap CI: resample samples, pool TP/FP/FN, compute micro F1 once per resample.
    For multi-annotator samples, randomly pick one annotator per draw.
    Returns (lower, upper) at the given alpha level (default 95% CI).
    """
    rng = np.random.default_rng(seed)
    n = len(group)
    boot_f1s = []
    for _ in range(n_boot):
        indices = rng.integers(0, n, size=n)
        tp = fp = fn = 0.0
        for i in indices:
            t, p, f = _sample_tpfpfn(group[i], rng, strict=strict)
            tp += t; fp += p; fn += f
        micro_p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        micro_r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)
                    if (micro_p + micro_r) > 0 else 0.0)
        boot_f1s.append(micro_f1)
    boot_f1s = np.array(boot_f1s)
    lo = np.percentile(boot_f1s, 100 * alpha / 2)
    hi = np.percentile(boot_f1s, 100 * (1 - alpha / 2))
    return float(np.mean(boot_f1s)), lo, hi


def bootstrap_macro_prf(group: list, n_boot: int = 1000, alpha: float = 0.05,
                        seed: int = 2023, strict: bool = False, acc_tp: bool = False,
                        exclude_acc: bool = False) -> dict:
    """
    Run bootstrap once, return mean+CI for P, R, F1 simultaneously (macro).
    Saves time vs calling bootstrap_ci three times separately.
    Returns dict: {metric: (mean, lo, hi)}

    If exclude_acc=True, samples where model correctly said all-correct (with no
    missed errors) are excluded from each bootstrap resample's mean.  If an
    entire resample contains no valid samples (unlikely), that draw is skipped.
    """
    rng = np.random.default_rng(seed)
    n = len(group)
    boot_p, boot_r, boot_f = [], [], []
    for _ in range(n_boot):
        indices = rng.integers(0, n, size=n)
        ps = [_sample_prf(group[i], 'precision', rng, strict=strict, acc_tp=acc_tp,
                          exclude_acc=exclude_acc) for i in indices]
        rs = [_sample_prf(group[i], 'recall',    rng, strict=strict, acc_tp=acc_tp,
                          exclude_acc=exclude_acc) for i in indices]
        fs = [_sample_prf(group[i], 'f1',        rng, strict=strict, acc_tp=acc_tp,
                          exclude_acc=exclude_acc) for i in indices]
        # filter out None values (excluded all-correct samples)
        ps = [v for v in ps if v is not None]
        rs = [v for v in rs if v is not None]
        fs = [v for v in fs if v is not None]
        if not fs:
            continue  # skip this degenerate draw
        boot_p.append(float(np.mean(ps)))
        boot_r.append(float(np.mean(rs)))
        boot_f.append(float(np.mean(fs)))
    result = {}
    for name, arr in [('precision', boot_p), ('recall', boot_r), ('f1', boot_f)]:
        a = np.array(arr)
        result[name] = (float(np.mean(a)),
                        float(np.percentile(a, 100 * alpha / 2)),
                        float(np.percentile(a, 100 * (1 - alpha / 2))))
    return result


def bootstrap_micro_prf(group: list, n_boot: int = 1000, alpha: float = 0.05,
                        seed: int = 2023, strict: bool = False, acc_tp: bool = False) -> dict:
    """
    Run bootstrap once, return mean+CI for micro P, R, F1 simultaneously.
    Returns dict: {metric: (mean, lo, hi)}
    """
    rng = np.random.default_rng(seed)
    n = len(group)
    boot_p, boot_r, boot_f = [], [], []
    for _ in range(n_boot):
        indices = rng.integers(0, n, size=n)
        tp = fp = fn = 0.0
        for i in indices:
            t, p, f = _sample_tpfpfn(group[i], rng, strict=strict, acc_tp=acc_tp)
            tp += t; fp += p; fn += f
        mp = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        mr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        mf = (2 * mp * mr / (mp + mr) if (mp + mr) > 0 else 0.0)
        boot_p.append(mp); boot_r.append(mr); boot_f.append(mf)
    result = {}
    for name, arr in [('precision', boot_p), ('recall', boot_r), ('f1', boot_f)]:
        a = np.array(arr)
        result[name] = (float(np.mean(a)),
                        float(np.percentile(a, 100 * alpha / 2)),
                        float(np.percentile(a, 100 * (1 - alpha / 2))))
    return result


def fmt(val, pct=False, signed=False):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return '  N/A '
    if pct:
        return f'{val*100:5.1f}%'
    if signed:
        return f'{val:+.3f}'
    return f'{val:.3f}'


def run_analysis(ground_truth: list):

    for d in ground_truth:
        d['gold_errors'] = compute_gold_errors(d)

    by_model = defaultdict(list)
    for d in ground_truth:
        by_model[d['model_name']].append(d)

    sizes = ['1', '2', '3', '4', '5', '6']
    cw = 8

    # Module 1: Overall P / R / F1
    # Two evaluation modes for wrong_type / wrong_triple predictions:
    #   Lenient: only FN  (model spotted the error triple, partial credit for detection)
    #   Strict : FP + FN  (CoNLL / seqeval convention, double penalty for mislabelling)
    # acc_tp=False: all-correct samples counted as P=1, R=1, TP=0 (no predictions made)
    # acc_tp=True : all-correct samples counted as TP=1 (correct all-correct judgment)
    print('\nModule 1: Overall Precision / Recall / F1')
    for strict, acc_tp in [
        (False, False),
    ]:
        # Macro: point estimate (all samples)
        print(f'\n    [Macro (all samples) -- mean of per-sample P/R/F1, each sample weighted equally]')
        print(f'    Confirmed all-correct samples with no missed errors contribute P=R=F1=1.0.')
        print(f'    {"Model":<8}{"N":>6}{"P":>8}{"R":>8}{"F1":>8}')
        print('    ' + '-'*38)
        for m in MODEL_ORDER:
            grp = by_model[m]
            res = analyse_group(grp, strict=strict, acc_tp=acc_tp)
            if res:
                print(f'    {MODEL_LABELS[m]:<8}{res["n"]:>6}'
                      f'{fmt(res["precision"]):>8}'
                      f'{fmt(res["recall"]):>8}'
                      f'{fmt(res["f1"]):>8}')

        # Macro: point estimate (error samples only -- exclude perfect all-correct)
        print(f'\n    [Macro (error samples only) -- exclude confirmed all-correct samples with no missed errors]')
        print(f'    Only samples where the model predicted errors OR the model said all-correct but missed real errors.')
        print(f'    N_used  = samples used in this average (predicted errors OR missed errors).')
        print(f'    N_skip  = perfect all-correct samples skipped (no missed errors).')
        print(f'    {"Model":<8}{"N_used":>8}{"N_skip":>8}{"P":>8}{"R":>8}{"F1":>8}')
        print('    ' + '-'*46)
        for m in MODEL_ORDER:
            grp = by_model[m]
            res = analyse_group(grp, strict=strict, acc_tp=acc_tp)
            if res:
                n_skip = res["n"] - res["n_excl"]
                print(f'    {MODEL_LABELS[m]:<8}{res["n_excl"]:>8}{n_skip:>8}'
                      f'{fmt(res["precision_excl"]):>8}'
                      f'{fmt(res["recall_excl"]):>8}'
                      f'{fmt(res["f1_excl"]):>8}')

        # Macro: bootstrap (all samples)
        print(f'\n    [Macro Bootstrap (all samples) -- BootP / BootR / BootF1 mean, F1 95% CI]')
        print(f'    {"Model":<8}{"BootP":>8}{"BootR":>8}{"BootF1":>8}  {"F1 95% CI":<16}')
        print('    ' + '-'*50)
        for m in MODEL_ORDER:
            grp = by_model[m]
            if grp:
                b = bootstrap_macro_prf(grp, strict=strict, acc_tp=acc_tp)
                pm = b['precision'][0]
                rm = b['recall'][0]
                fm, flo, fhi = b['f1']
                ci_str = f'[{flo:.3f}, {fhi:.3f}]'
                print(f'    {MODEL_LABELS[m]:<8}'
                      f'{pm:>8.3f}{rm:>8.3f}{fm:>8.3f}  {ci_str:<16}')

        # Macro: bootstrap (error samples only)
        print(f'\n    [Macro Bootstrap (error samples only) -- BootP / BootR / BootF1 mean, F1 95% CI]')
        print(f'    {"Model":<8}{"BootP":>8}{"BootR":>8}{"BootF1":>8}  {"F1 95% CI":<16}')
        print('    ' + '-'*50)
        for m in MODEL_ORDER:
            grp = by_model[m]
            if grp:
                b = bootstrap_macro_prf(grp, strict=strict, acc_tp=acc_tp, exclude_acc=True)
                pm = b['precision'][0]
                rm = b['recall'][0]
                fm, flo, fhi = b['f1']
                ci_str = f'[{flo:.3f}, {fhi:.3f}]'
                print(f'    {MODEL_LABELS[m]:<8}'
                      f'{pm:>8.3f}{rm:>8.3f}{fm:>8.3f}  {ci_str:<16}')

        # Micro: point estimate
        print(f'\n    [Micro  -- pool TP/FP/FN across all samples, errors weighted by count]')
        print(f'    {"Model":<8}{"N":>6}{"P":>8}{"R":>8}{"F1":>8}')
        print('    ' + '-'*38)
        for m in MODEL_ORDER:
            grp = by_model[m]
            res = analyse_group(grp, strict=strict, acc_tp=acc_tp)
            if res:
                print(f'    {MODEL_LABELS[m]:<8}{res["n"]:>6}'
                      f'{fmt(res["micro_precision"]):>8}'
                      f'{fmt(res["micro_recall"]):>8}'
                      f'{fmt(res["micro_f1"]):>8}')

        # Micro: bootstrap
        print(f'\n    [Micro Bootstrap -- BootP / BootR / BootF1 mean, F1 95% CI]')
        print(f'    {"Model":<8}{"BootP":>8}{"BootR":>8}{"BootF1":>8}  {"F1 95% CI":<16}')
        print('    ' + '-'*50)
        for m in MODEL_ORDER:
            grp = by_model[m]
            if grp:
                b = bootstrap_micro_prf(grp, strict=strict, acc_tp=acc_tp)
                pm = b['precision'][0]
                rm = b['recall'][0]
                fm, flo, fhi = b['f1']
                ci_str = f'[{flo:.3f}, {fhi:.3f}]'
                print(f'    {MODEL_LABELS[m]:<8}'
                      f'{pm:>8.3f}{rm:>8.3f}{fm:>8.3f}  {ci_str:<16}')

        # Sample-level binary classification of all-correct judgment
        print(f'\n    [Sample-Level Binary Classification of All-Correct Judgment]')
        print(f'    Positive class = "all correct" (model predicted no errors); Negative class = "has errors".')
        print(f'    TP = model said all-correct AND gold has no errors   (correct abstention).')
        print(f'    FP = model said all-correct BUT gold has errors      (dangerous miss: silent failure).')
        print(f'    FN = model predicted errors BUT gold has no errors   (false alarm at sample level).')
        print(f'    TN = model predicted errors AND gold has errors      (correctly flagged as erroneous).')
        print(f'    Bin-P   = TP / (TP+FP): among samples where model said all-correct, fraction truly all-correct.')
        print(f'    Bin-R   = TP / (TP+FN): among gold all-correct samples, fraction model correctly identified.')
        print(f'    Bin-F1  = harmonic mean of Bin-P and Bin-R.')
        print(f'    Bin-Acc = (TP+TN) / N: fraction of all samples where the model\'s judgment was correct.')
        print()
        cw3 = 8
        print(f'    {"Model":<8}'
              f'{"TP":>{cw3}}{"FP":>{cw3}}{"FN":>{cw3}}{"TN":>{cw3}}'
              f'{"Bin-P":>{cw3}}{"Bin-R":>{cw3}}{"Bin-F1":>{cw3}}{"Bin-Acc":>{cw3}}')
        print('    ' + '-' * (8 + cw3 * 8))
        for m in MODEL_ORDER:
            res = analyse_group(by_model[m], strict=strict, acc_tp=acc_tp)
            if not res:
                continue
            print(f'    {MODEL_LABELS[m]:<8}'
                  f'{res["bin_tp"]:>{cw3}}{res["bin_fp"]:>{cw3}}'
                  f'{res["bin_fn"]:>{cw3}}{res["bin_tn"]:>{cw3}}'
                  f'{fmt(res["bin_precision"], pct=True):>{cw3}}'
                  f'{fmt(res["bin_recall"],    pct=True):>{cw3}}'
                  f'{fmt(res["bin_f1"],        pct=True):>{cw3}}'
                  f'{fmt(res["bin_accuracy"],  pct=True):>{cw3}}')

    # Module 2: Error Pattern Analysis
    print('\nModule 2: Error Pattern Analysis')
    print('  (Breakdown of how the model\'s predictions are wrong, based on human judgement)\n')

    # Table 2a: denom = total model predictions
    print('  [Table 2a: Error Rates (denom = total model predictions)]')
    print('  OverPred%  = % of predicted errors the human says do not actually exist (false alarms).')
    print('  WrongTyp%  = % of predicted errors where the triple is right but the error type is wrong.')
    print('  WrongTrp%  = % of predicted errors where the type is right but the triple is wrong.')
    print('  MissRate%  = % of samples where the model missed at least one real error (denom = samples).')
    print('  AvgMiss    = avg. number of real errors missed per sample, incl. samples with 0 missed (denom = samples).')
    print()
    cw2 = 10
    print(f'  {"Model":<8}'
          f'{"OverPred%":>{cw2}}'
          f'{"WrongTyp%":>{cw2}}'
          f'{"WrongTrp%":>{cw2}}'
          f'{"MissRate%":>{cw2}}'
          f'{"AvgMiss":>{cw2}}')
    print('  ' + '-' * (8 + cw2 * 5))
    for m in MODEL_ORDER:
        res = analyse_group(by_model[m])
        if not res:
            continue
        print(f'  {MODEL_LABELS[m]:<8}'
              f'{fmt(res["over_pred_rate"],  pct=True):>{cw2}}'
              f'{fmt(res["wrong_type_rate"], pct=True):>{cw2}}'
              f'{fmt(res["wrong_trip_rate"], pct=True):>{cw2}}'
              f'{fmt(res["miss_rate"],       pct=True):>{cw2}}'
              f'{res["avg_missed"]:>{cw2}.2f}')

    # Table 2b: denom = total mistakes (over-pred + wrong type + wrong triple + missed)
    print()
    print('  [Table 2b: Error Type Breakdown (denom = total mistakes)]')
    print('  Mistakes = over-predictions + wrong-type predictions + wrong-triple predictions + missed errors.')
    print('  Shows how the model\'s mistakes are distributed across error categories; the four columns sum to 100%.')
    print('  OverPred%  = % of all mistakes that are false alarms (model predicted a non-existent error).')
    print('  WrongTyp%  = % of all mistakes where the model found the right triple but labelled it with the wrong type.')
    print('  WrongTrp%  = % of all mistakes where the model used the right type but pointed to the wrong triple.')
    print('  Missed%    = % of all mistakes that are real errors the model completely failed to predict.')
    print()
    print(f'  {"Model":<8}'
          f'{"OverPred%":>{cw2}}'
          f'{"WrongTyp%":>{cw2}}'
          f'{"WrongTrp%":>{cw2}}'
          f'{"Missed%":>{cw2}}')
    print('  ' + '-' * (8 + cw2 * 4))
    for m in MODEL_ORDER:
        res = analyse_group(by_model[m])
        if not res:
            continue
        print(f'  {MODEL_LABELS[m]:<8}'
              f'{fmt(res["over_rate_m"],  pct=True):>{cw2}}'
              f'{fmt(res["wtype_rate_m"], pct=True):>{cw2}}'
              f'{fmt(res["wtrip_rate_m"], pct=True):>{cw2}}'
              f'{fmt(res["miss_rate_m"],  pct=True):>{cw2}}')

    # Module 3: F1 by Triple Count (size)
    print('\nModule 3: F1 by Triple Count (size)')
    for strict, acc_tp in [
        (False, False),
    ]:
        for averaging, f1key, pkey, rkey in [
            ('Macro', 'f1',       'precision',       'recall'),
            ('Micro', 'micro_f1', 'micro_precision', 'micro_recall'),
        ]:
            print(f'\n    [{averaging} F1]')
            print(f'    {"Size":<6}' + ''.join(f'{MODEL_LABELS[m]:>{cw}}' for m in MODEL_ORDER)
                  + f'{"N":>{cw}}')
            print('    ' + '-' * (6 + cw * (len(MODEL_ORDER) + 1)))
            for sz in sizes:
                row, total_n = [], 0
                for m in MODEL_ORDER:
                    grp = [d for d in by_model[m] if d['size'] == sz]
                    res = analyse_group(grp, strict=strict, acc_tp=acc_tp)
                    row.append(f'{res[f1key]:.3f}' if res else '-')
                    total_n += res['n'] if res else 0
                print(f'    {sz:<6}' + ''.join(f'{v:>{cw}}' for v in row) + f'{total_n:>{cw}}')

            for metric_label, mkey in [(f'{averaging} P', pkey), (f'{averaging} R', rkey)]:
                print(f'\n      [{metric_label}]')
                print(f'      {"Size":<6}' + ''.join(f'{MODEL_LABELS[m]:>{cw}}' for m in MODEL_ORDER))
                print('      ' + '-' * (4 + cw * len(MODEL_ORDER)))
                for sz in sizes:
                    row = []
                    for m in MODEL_ORDER:
                        grp = [d for d in by_model[m] if d['size'] == sz]
                        res = analyse_group(grp, strict=strict, acc_tp=acc_tp)
                        row.append(f'{res[mkey]:.3f}' if res else '-')
                    print(f'      {sz:<6}' + ''.join(f'{v:>{cw}}' for v in row))

    # Module 4: F1 by Gold Error Count (error density)
    # Use threshold bucketing to handle averaged float gold counts from multi-annotator samples.
    # < 0.5  -> bucket 0;  0.5 <= x < 1.5 -> bucket 1;  >= 1.5 -> bucket 2+
    buckets = [('0  (all correct)', lambda d: d['gold_errors']['total'] < 0.5),
               ('1',               lambda d: 0.5 <= d['gold_errors']['total'] < 1.5),
               ('2',               lambda d: 1.5 <= d['gold_errors']['total'] < 2.5),
               ('>=3',             lambda d: d['gold_errors']['total'] >= 2.5)]

    print('\nModule 4: F1 by Gold Error Count (error density)')

    # Gold error distribution (overall and per model)
    total_all = len(ground_truth)
    dist_rows = [
        ('0  (all correct)', lambda d: d['gold_errors']['total'] < 0.5),
        ('1',                lambda d: 0.5 <= d['gold_errors']['total'] < 1.5),
        ('2',                lambda d: 1.5 <= d['gold_errors']['total'] < 2.5),
        ('3',                lambda d: 2.5 <= d['gold_errors']['total'] < 3.5),
        ('4',                lambda d: d['gold_errors']['total'] >= 3.5),
    ]

    print('\n  [Gold Error Distribution]')
    print(f'  {"GoldErr":<18}' + ''.join(f'{MODEL_LABELS[m]:>{cw}}' for m in MODEL_ORDER) + f'{"Total":>{cw}}')
    print('  ' + '-' * (18 + cw * (len(MODEL_ORDER) + 1)))
    for label, fn in dist_rows:
        row = [sum(1 for d in by_model[m] if fn(d)) for m in MODEL_ORDER]
        print(f'  {label:<18}' + ''.join(f'{v:>{cw}}' for v in row) + f'{sum(row):>{cw}}')
    print(f'  {"Total":<18}' + ''.join(f'{len(by_model[m]):>{cw}}' for m in MODEL_ORDER) + f'{total_all:>{cw}}')
    print()
    print(f'  {"GoldErr (%)":18}' + ''.join(f'{MODEL_LABELS[m]:>{cw}}' for m in MODEL_ORDER) + f'{"Overall":>{cw}}')
    print('  ' + '-' * (18 + cw * (len(MODEL_ORDER) + 1)))
    ns = [len(by_model[m]) for m in MODEL_ORDER]
    for label, fn in dist_rows:
        row = [sum(1 for d in by_model[m] if fn(d)) for m in MODEL_ORDER]
        pcts = [f'{v/n*100:.1f}%' if n > 0 else 'N/A' for v, n in zip(row, ns)]
        overall_pct = f'{sum(row)/total_all*100:.1f}%'
        print(f'  {label:<18}' + ''.join(f'{v:>{cw}}' for v in pcts) + f'{overall_pct:>{cw}}')
    for strict, acc_tp in [
        (False, False),
    ]:
        for averaging, f1key in [('Macro', 'f1'), ('Micro', 'micro_f1')]:
            print(f'\n    [{averaging} F1]')
            print(f'    {"GoldErr":<18}' + ''.join(f'{MODEL_LABELS[m]:>{cw}}' for m in MODEL_ORDER)
                  + f'{"N":>{cw}}')
            print('    ' + '-' * (18 + cw * (len(MODEL_ORDER) + 1)))
            for label, fn in buckets:
                row, total_n = [], 0
                for m in MODEL_ORDER:
                    grp = [d for d in by_model[m] if fn(d)]
                    res = analyse_group(grp, strict=strict, acc_tp=acc_tp)
                    row.append(f'{res[f1key]:.3f}' if res else '-')
                    total_n += res['n'] if res else 0
                print(f'    {label:<18}' + ''.join(f'{v:>{cw}}' for v in row) + f'{total_n:>{cw}}')

    print(f'\n  [Sample counts per bucket]')
    print(f'  {"GoldErr":<18}' + ''.join(f'{MODEL_LABELS[m]:>{cw}}' for m in MODEL_ORDER))
    print('  ' + '-' * (18 + cw * len(MODEL_ORDER)))
    for label, fn in buckets:
        row = [str(sum(1 for d in by_model[m] if fn(d))) for m in MODEL_ORDER]
        print(f'  {label:<18}' + ''.join(f'{v:>{cw}}' for v in row))

    def pct(a, t): return f'{a/t*100:.1f}%' if t > 0 else 'N/A'

    # Module 5: Error Type Distribution
    print('\nModule 5: Error Type Distribution (model predictions vs. human supplements)')
    for m in MODEL_ORDER:
        grp = by_model[m]
        label = MODEL_LABELS[m]

        pred_miss  = sum(len(d['parsed_errors']['missing'])   for d in grp)
        pred_extra = sum(len(d['parsed_errors']['extra'])     for d in grp)
        pred_incr  = sum(len(d['parsed_errors']['incorrect']) for d in grp)
        pred_total = pred_miss + pred_extra + pred_incr

        miss_miss = miss_extra = miss_incr = 0.0
        for d in grp:
            anns = d.get('all_annotations') or ([d['human_annotation']] if d['human_annotation'] else [])
            for ann in anns:
                me = ann.get('missed_errors', {})
                miss_miss  += len(me.get('missing', []))   / len(anns)
                miss_extra += len(me.get('extra', []))     / len(anns)
                miss_incr  += len(me.get('incorrect', [])) / len(anns)
        miss_total = miss_miss + miss_extra + miss_incr

        print(f'\n  {label}:')
        print(f'    {"":22} {"Missing":>10} {"Extra":>10} {"Incorrect":>10} {"Total":>8}')
        print(f'    {"Model predicted":22} {pred_miss:>10} {pred_extra:>10} {pred_incr:>10} {pred_total:>8}')
        print(f'    {"  (%)":22} {pct(pred_miss,pred_total):>10} '
              f'{pct(pred_extra,pred_total):>10} {pct(pred_incr,pred_total):>10}')
        print(f'    {"Human supplemented":22} {miss_miss:>10.1f} {miss_extra:>10.1f} {miss_incr:>10.1f} {miss_total:>8.1f}')
        print(f'    {"  (%)":22} {pct(miss_miss,miss_total):>10} '
              f'{pct(miss_extra,miss_total):>10} {pct(miss_incr,miss_total):>10}')

    # Module 5b: Per-Error-Type Micro P/R/F1
    # For each error type, pool TP/FP/FN across all samples and compute micro P/R/F1.
    # TP_t: model predicted type t AND human confirms the triple is a real error of type t (is_correct).
    # FP_t: model predicted type t BUT it is wrong (over-prediction, wrong triple, or wrong type).
    # FN_t: gold errors of type t that the model did not correctly predict (= gold_t - TP_t).
    # This uses lenient matching: wrong_type counts as FP for predicted type but does NOT give
    # partial TP credit (unlike the overall P/R/F1 where wrong_type/wrong_triple gives partial credit).
    print('\nModule 5b: Per-Error-Type Micro P/R/F1')
    print('  Micro P/R/F1 computed by pooling TP/FP/FN across all samples per model.')
    print('  TP = model predicted type t correctly (is_correct with matching predicted_type).')
    print('  FP = model predicted type t but it is wrong (over-prediction, wrong triple, or wrong type label).')
    print('  FN = gold errors of type t not correctly predicted by the model (gold_t - TP_t).')
    print()
    errtypes = ['missing', 'extra', 'incorrect']
    cw5 = 10
    # Header
    header = f'  {"Model":<8}'
    for et in errtypes:
        header += f'  {et.capitalize():^26}'
    print(header)
    subhdr = f'  {"":8}'
    for _ in errtypes:
        subhdr += f'  {"P":>{cw5}}{"R":>{cw5}}{"F1":>{cw5}}'
    print(subhdr)
    print('  ' + '-' * (8 + (cw5 * 3 + 2) * 3))

    for m in MODEL_ORDER:
        grp = by_model[m]
        # Pool TP/FP/FN across all samples for each type
        tp_sum = {'missing': 0.0, 'extra': 0.0, 'incorrect': 0.0}
        fp_sum = {'missing': 0.0, 'extra': 0.0, 'incorrect': 0.0}
        fn_sum = {'missing': 0.0, 'extra': 0.0, 'incorrect': 0.0}
        for d in grp:
            res = compute_typewise_tpfpfn(d)
            for t in errtypes:
                tp_sum[t] += res['tp'][t]
                fp_sum[t] += res['fp'][t]
                fn_sum[t] += res['fn'][t]
        row = f'  {MODEL_LABELS[m]:<8}'
        for t in errtypes:
            tp, fp, fn = tp_sum[t], fp_sum[t], fn_sum[t]
            p  = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
            r  = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
            f1 = (2*p*r/(p+r) if (not np.isnan(p) and not np.isnan(r) and (p+r) > 0)
                  else float('nan'))
            row += f'  {fmt(p, pct=True):>{cw5}}{fmt(r, pct=True):>{cw5}}{fmt(f1, pct=True):>{cw5}}'
        print(row)


if __name__ == '__main__':
    log_path = BASE / 'analysis_output.log'
    tee = _Tee(log_path)
    sys.stdout = tee
    try:
        ground_truth = build_ground_truth()
        run_analysis(ground_truth)
        print('\nLog saved.')
    finally:
        tee.close()
