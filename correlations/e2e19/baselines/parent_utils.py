"""
parent_utils.py
---------------
Pure-Python PARENT metric functions extracted from Google's parent.py.
No TensorFlow dependency — only stdlib (collections, math).

Reference: Dhingra et al., "Handling Divergent Reference Texts when Evaluating
Table-to-Text Generation", ACL 2019.
https://github.com/google-research/language/tree/master/language/table_text_eval

Usage
-----
from parent_utils import parent

predictions = [["the", "blue", "spice", "is", "a", "coffee", "shop"]]
references  = [[["the", "blue", "spice", "is", "a", "coffee", "shop", "in", "city", "centre"]]]
tables      = [[(["blue", "spice"], ["eat", "type"], ["coffee", "shop"]),
                (["blue", "spice"], ["area"],         ["city", "centre"])]]

avg_p, avg_r, avg_f, all_f = parent(predictions, references, tables)
# all_f is a list of per-sample F1 scores
"""

import collections
import math


# ── Entailment function ──────────────────────────────────────────────────────

def overlap_probability(ngram, table, smoothing=0.0, stopwords=None):
    """Probability that ngram overlaps with table token values."""
    if len(table[0]) == 2:
        table_values = set(tok for _, value in table for tok in value)
    else:
        table_values = set(tok for head, _, tail in table for tok in head + tail)
    overlap = 0
    for token in ngram:
        if stopwords is not None and token in stopwords:
            overlap += 1
            continue
        if token in table_values:
            overlap += 1
    return float(overlap + smoothing) / float(len(ngram) + smoothing)


# ── Mention function ─────────────────────────────────────────────────────────

def _len_lcs(x, y):
    table = _lcs(x, y)
    return table[len(x), len(y)]


def _lcs(x, y):
    n, m = len(x), len(y)
    table = {}
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 or j == 0:
                table[i, j] = 0
            elif x[i - 1] == y[j - 1]:
                table[i, j] = table[i - 1, j - 1] + 1
            else:
                table[i, j] = max(table[i - 1, j], table[i, j - 1])
    return table


def _mention_probability(table_entry, sentence, smoothing=0.0):
    """Probability that the table entry is mentioned in the sentence (via LCS)."""
    if len(table_entry) == 2:
        value = table_entry[1]
    else:
        value = table_entry[0] + table_entry[2]
    overlap = _len_lcs(value, sentence)
    return float(overlap + smoothing) / float(len(value) + smoothing)


# ── N-gram helpers ────────────────────────────────────────────────────────────

def _ngrams(sequence, order):
    assert order >= 1
    for n in range(order, len(sequence) + 1):
        yield tuple(sequence[n - order: n])


def _ngram_counts(sequence, order):
    if len(sequence) < order:
        return collections.Counter()
    return collections.Counter(_ngrams(sequence, order))


# ── PARENT ───────────────────────────────────────────────────────────────────

def parent(predictions,
           references,
           tables,
           lambda_weight=0.5,
           smoothing=0.00001,
           max_order=4,
           entailment_fn=overlap_probability,
           mention_fn=_mention_probability):
    """
    PARENT metric.

    Args:
        predictions  : List[List[str]]         — tokenized generated texts
        references   : List[List[List[str]]]   — per-sample list of tokenized refs
        tables       : List[List[Tuple]]        — per-sample list of triples.
                       Each triple is (head_tokens, relation_tokens, tail_tokens),
                       where each member is a List[str].
                       E.g. [(['blue','spice'], ['eat','type'], ['coffee','shop'])]
        lambda_weight: float in [0,1], weight for table recall vs reference recall.
        smoothing    : float, Laplace smoothing value.
        max_order    : int, max n-gram order.

    Returns:
        avg_precision : float
        avg_recall    : float
        avg_f1        : float
        all_f_scores  : List[float]  — per-sample F1 (same order as predictions)
    """
    precisions, recalls, all_f_scores = [], [], []

    for prediction, list_of_references, table in zip(predictions, references, tables):
        c_prec, c_rec, c_f = [], [], []
        ref_rec, table_rec = [], []

        for reference in list_of_references:
            ngram_prec, ngram_rec = [], []

            for order in range(1, max_order + 1):
                pred_ngram_counts = _ngram_counts(prediction, order)
                pred_ngram_weights = {ng: entailment_fn(ng, table)
                                      for ng in pred_ngram_counts}
                ref_ngram_counts = _ngram_counts(reference, order)
                ref_ngram_weights = {ng: entailment_fn(ng, table)
                                     for ng in ref_ngram_counts}

                # Precision
                num, den = 0., 0.
                for ng, cnt in pred_ngram_counts.items():
                    den += cnt
                    p_in_ref = min(1., float(ref_ngram_counts.get(ng, 0) / cnt))
                    num += cnt * (p_in_ref + (1. - p_in_ref) * pred_ngram_weights[ng])
                ngram_prec.append(0.0 if den == 0. else num / den)

                # Recall
                num, den = 0., 0.
                for ng, cnt in ref_ngram_counts.items():
                    p_in_pred = min(1., float(pred_ngram_counts.get(ng, 0) / cnt))
                    den += cnt * ref_ngram_weights[ng]
                    num += cnt * ref_ngram_weights[ng] * p_in_pred
                ngram_rec.append(1.0 if den == 0. else num / den)

            # Table recall
            table_mention = [mention_fn(entry, prediction) for entry in table]
            table_rec.append(sum(table_mention) / len(table))

            # Smoothing for orders > 1
            for order in range(1, max_order):
                if ngram_prec[order] == 0.:
                    ngram_prec[order] = smoothing
                if ngram_rec[order] == 0.:
                    ngram_rec[order] = smoothing

            # Geometric average precision
            w = 1. / max_order
            if any(p == 0. for p in ngram_prec):
                c_prec.append(0.)
            else:
                c_prec.append(math.exp(math.fsum(w * math.log(p) for p in ngram_prec)))

            # Geometric average reference recall
            if any(r == 0. for r in ngram_rec):
                ref_rec.append(smoothing)
            else:
                ref_rec.append(math.exp(math.fsum(w * math.log(r) for r in ngram_rec)))

            # Combined recall
            tr = table_rec[-1] or smoothing
            rr = ref_rec[-1]
            if rr == 0. or tr == 0.:
                c_rec.append(0.)
            else:
                lw = lambda_weight
                c_rec.append(math.exp((1. - lw) * math.log(rr) + lw * math.log(tr)))

            # F1
            p_, r_ = c_prec[-1], c_rec[-1]
            c_f.append((2. * p_ * r_) / (p_ + r_ + 1e-8))

        # Best reference
        best = max(range(len(c_f)), key=lambda i: c_f[i])
        precisions.append(c_prec[best])
        recalls.append(c_rec[best])
        all_f_scores.append(c_f[best])

    avg_p = sum(precisions) / len(precisions)
    avg_r = sum(recalls) / len(recalls)
    avg_f = sum(all_f_scores) / len(all_f_scores)
    return avg_p, avg_r, avg_f, all_f_scores
