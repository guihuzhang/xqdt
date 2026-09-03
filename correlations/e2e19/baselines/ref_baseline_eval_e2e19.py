#!/usr/bin/env python3
"""
ref_baseline_eval_e2e19.py

Reference-based baseline evaluation on E2E 2019 test set.
Metrics: BLEU · METEOR · PARENT · BERTScore · BARTScore · BLEURT

Two-stage pipeline (mirrors baseline_eval_e2e19.py):
  Stage 1  Compute & cache per-sample scores  →  ref_baseline_scores_e2e19.json
  Stage 2  Apply worker filtering, compute text/system-level correlations

Reference source : data/e2e/refs_e2e19.json   (refs_2019 ∪ refs_2017 per MR)
Human quality    : filtered fine_score / 100   (strategy: drop_gt100_all + A+C+D+E+F)

Downloads needed (place in this directory, baselines/):
  bart_score.pth    BARTScore fine-tuned weights
  BLEURT-20/        BLEURT checkpoint directory

Required packages:
  sacrebleu  nltk  bert-score  torch  scipy  tqdm  transformers
  tensorflow (for BLEURT)
  (bart_score.py and bleurt/ are already in this directory)
"""

import json
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
from tqdm import tqdm

# Paths
THIS_DIR   = Path(__file__).resolve().parent            # baselines/
BASE       = THIS_DIR.parent                            # err_pred_e2e19/
CONVERTED  = BASE / 'human_ratings' / 'converted.json'
REFS_JSON  = BASE.parent / 'e2e' / 'refs_e2e19.json'
RESULTS_DIR = BASE / 'llm_evaluation_results'
LOG_FILE   = RESULTS_DIR / 'ref_baseline_eval_e2e19.log'
CACHE_FILE = RESULTS_DIR / 'ref_baseline_scores_e2e19.json'

sys.path.insert(0, str(THIS_DIR))   # make bart_score, bleurt importable
sys.path.insert(0, str(BASE))

from worker_filter import load_excluded_workers

# Metrics to run
USE_BLEU      = True
USE_METEOR    = True
USE_PARENT    = True
USE_BERTSCORE = True
USE_BARTSCORE = True   # needs bart_score.pth in THIS_DIR
USE_BLEURT    = True   # needs BLEURT-20/ in THIS_DIR

BART_SCORE_PATH   = str(THIS_DIR / 'bart_score.pth')
BLEURT_CHECKPOINT = str(THIS_DIR / 'BLEURT-20')
BERTSCORE_BATCH   = 16
BARTSCORE_BATCH   = 16   # increase if GPU/MPS memory allows


# Device helper (CUDA/MPS/CPU)
def get_device() -> str:
    import torch
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


# Worker filtering (same as baseline_eval_e2e19.py)
EXCLUDED_WORKERS = load_excluded_workers()


# Data helpers

def normalize_mr(mr_string: str) -> str:
    """Sort MR slots alphabetically — matches e2e_csv2json2019.py."""
    parts = [p.strip() for p in mr_string.split(',') if p.strip()]
    return ', '.join(sorted(parts))


def mr_to_parent_triples(mr_text: str):
    """
    Parse E2E MR text → list of (head_toks, pred_toks, tail_toks) for PARENT.
    All tokens are lowercase strings. The 'name' slot defines the subject
    and is not emitted as a triple.
    """
    slot_pairs = [
        (m.group(1).strip(), m.group(2).strip())
        for m in re.finditer(r'\s*([^,\[]+?)\s*\[([^\]]*)\]\s*(?:,|$)', mr_text)
    ]
    subject = next(
        (val for slot, val in slot_pairs if slot.lower() == 'name' and val),
        'restaurant',
    )
    return [
        (subject.lower().split(), slot.lower().split(), val.lower().split())
        for slot, val in slot_pairs
        if slot.lower() != 'name' and val
    ]


def get_refs(mr_text: str, refs_lookup: dict) -> list:
    """Return deduplicated refs (2019 ∪ 2017) for this MR."""
    entry = refs_lookup.get(normalize_mr(mr_text), {})
    seen, combined = set(), []
    for r in entry.get('refs_2019', []) + entry.get('refs_2017', []):
        if r not in seen:
            seen.add(r)
            combined.append(r)
    return combined


def _ensure_nltk_meteor_resources():
    import nltk
    for corpus_name in ('wordnet', 'omw-1.4'):
        try:
            nltk.data.find(f'corpora/{corpus_name}')
        except LookupError:
            nltk.download(corpus_name, quiet=True)


def load_all_samples(converted: dict) -> list:
    samples = []
    for mr_id, mr_data in converted.items():
        mr_text = mr_data['mr']
        for sys_name, sys_data in mr_data.get('systems', {}).items():
            samples.append({
                'mr_id':    mr_id,
                'sys_name': sys_name,
                'mr':       mr_text,
                'output':   sys_data.get('output', ''),
            })
    return samples


def add_human_quality(samples: list, converted: dict,
                      excluded_workers=EXCLUDED_WORKERS, drop_gt100=True):
    """Attach filtered human_quality (float 0–1 or None) in place."""
    for s in samples:
        annots = (converted
                  .get(str(s['mr_id']), {})
                  .get('systems', {})
                  .get(s['sys_name'], {})
                  .get('quality', []))
        scores = [
            a['fine_score'] for a in annots
            if str(a['worker_id']) not in excluded_workers
            and not (drop_gt100 and a['fine_score'] > 100)
        ]
        s['human_quality'] = (float(np.mean(scores)) / 100.0) if scores else None


# Correlation (same as baseline_eval_e2e19.py)

def _safe_corr(fn, x, y):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            v = fn(x, y)[0]
    except Exception:
        return float('nan')
    return float(v) if v is not None and np.isfinite(float(v)) else float('nan')


def _safe_mean(vals):
    arr = np.array([v for v in vals if np.isfinite(v)], dtype=float)
    return float(np.mean(arr)) if arr.size else float('nan')


def compute_correlations(samples: list, metric_key: str):
    """Text-level (per-MR avg) + system-level Pearson/Spearman/Kendall."""
    mr_data   = defaultdict(lambda: {'m': [], 'q': []})
    sys_model = defaultdict(list)
    sys_human = defaultdict(list)
    n_valid   = 0

    for s in samples:
        m = s.get(metric_key)
        q = s.get('human_quality')
        if m is None or q is None:
            continue
        try:
            m, q = float(m), float(q)
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(m) and np.isfinite(q)):
            continue
        n_valid += 1
        mr_data[s['mr_id']]['m'].append(m)
        mr_data[s['mr_id']]['q'].append(q)
        sys_model[s['sys_name']].append(m)
        sys_human[s['sys_name']].append(q)

    # Text-level: per-MR correlation, averaged
    tp, ts, tk = [], [], []
    for mid, d in mr_data.items():
        if len(d['m']) < 2:
            continue
        x, y = np.array(d['m']), np.array(d['q'])
        tp.append(_safe_corr(pearsonr,   x, y))
        ts.append(_safe_corr(spearmanr,  x, y))
        tk.append(_safe_corr(kendalltau, x, y))
    text = (_safe_mean(tp), _safe_mean(ts), _safe_mean(tk))

    # System-level: one point per system
    sys_names = sorted(sys_model.keys())
    if len(sys_names) > 1:
        sx = np.array([np.mean(sys_model[s]) for s in sys_names])
        sy = np.array([np.mean(sys_human[s]) for s in sys_names])
        system = (
            _safe_corr(pearsonr,   sx, sy),
            _safe_corr(spearmanr,  sx, sy),
            _safe_corr(kendalltau, sx, sy),
        )
    else:
        system = (float('nan'), float('nan'), float('nan'))

    return text, system, n_valid


# Metric computation functions

def compute_bleu(samples: list, refs_lookup: dict) -> list:
    """sacrebleu sentence_bleu, score in [0, 1]. Returns List[float|None]."""
    from sacrebleu.metrics import BLEU
    bleu = BLEU()
    out = []
    for s in tqdm(samples, desc='BLEU'):
        refs = s.get('refs')
        if refs is None:
            refs = get_refs(s['mr'], refs_lookup)
        if not refs:
            out.append(None)
            continue
        out.append(bleu.sentence_score(s['output'], refs, effective_order=True).score / 100.0)
    return out


def compute_meteor(samples: list, refs_lookup: dict) -> list:
    """NLTK METEOR with multiple references. Returns List[float|None]."""
    from nltk.translate.meteor_score import meteor_score
    _ensure_nltk_meteor_resources()
    out = []
    for s in tqdm(samples, desc='METEOR'):
        refs = s.get('refs')
        if refs is None:
            refs = get_refs(s['mr'], refs_lookup)
        if not refs:
            out.append(None)
            continue
        hyp_toks  = s['output'].lower().split()
        refs_toks = [r.lower().split() for r in refs]
        out.append(float(meteor_score(refs_toks, hyp_toks)))
    return out


def compute_parent(samples: list, refs_lookup: dict) -> list:
    """PARENT F1 (λ=0.5, pure Python). Returns List[float|None]."""
    from parent_utils import parent as parent_fn

    valid_idx = []
    for i, s in enumerate(samples):
        refs = s.get('refs')
        if refs is None:
            refs = get_refs(s['mr'], refs_lookup)
        if refs:
            valid_idx.append(i)
    if not valid_idx:
        return [None] * len(samples)

    all_preds, all_refs_tok, all_tables = [], [], []
    for i in tqdm(valid_idx, desc='PARENT (build)'):
        s = samples[i]
        refs = s.get('refs')
        if refs is None:
            refs = get_refs(s['mr'], refs_lookup)
        pred    = s['output'].lower().split()
        rtoks   = [r.lower().split() for r in refs]
        triples = mr_to_parent_triples(s['mr'])
        if not triples:  # MR has only a name slot — rare edge case
            triples = [(['restaurant'], ['type'], ['restaurant'])]
        all_preds.append(pred)
        all_refs_tok.append(rtoks)
        all_tables.append(triples)

    print(f'  PARENT: scoring {len(valid_idx)}/{len(samples)} samples')
    _, _, _, f_scores = parent_fn(all_preds, all_refs_tok, all_tables)

    result = [None] * len(samples)
    for idx, f in zip(valid_idx, f_scores):
        result[idx] = float(f)
    return result


def compute_bertscore(samples: list, refs_lookup: dict) -> list:
    """BERTScore F1 (lang=en). Returns List[float|None]."""
    from bert_score import score as bs_score

    valid_idx = []
    for i, s in enumerate(samples):
        refs = s.get('refs')
        if refs is None:
            refs = get_refs(s['mr'], refs_lookup)
        if refs:
            valid_idx.append(i)
    if not valid_idx:
        return [None] * len(samples)

    cands = [samples[i]['output'] for i in valid_idx]
    refs = []
    for i in valid_idx:
        rs = samples[i].get('refs')
        if rs is None:
            rs = get_refs(samples[i]['mr'], refs_lookup)
        refs.append(rs)

    print(f'  BERTScore: scoring {len(cands)} samples (lang=en, batch={BERTSCORE_BATCH})')
    result = [None] * len(samples)

    try:
        _, _, F1 = bs_score(cands, refs, lang='en', verbose=True, batch_size=BERTSCORE_BATCH)
        for k, idx in enumerate(valid_idx):
            result[idx] = float(F1[k].item())
        return result
    except Exception as exc:
        print(f'  BERTScore: multi-reference mode failed ({exc}); fallback to pairwise max-over-refs')

    flat_cands, flat_refs, owners = [], [], []
    for local_i, ref_list in enumerate(refs):
        for ref in ref_list:
            flat_cands.append(cands[local_i])
            flat_refs.append(ref)
            owners.append(local_i)

    if not flat_cands:
        return result

    _, _, flat_f1 = bs_score(flat_cands, flat_refs, lang='en', verbose=True, batch_size=BERTSCORE_BATCH)
    bucket = defaultdict(list)
    for local_i, sc in zip(owners, flat_f1):
        bucket[local_i].append(float(sc.item()))

    for local_i, idx in enumerate(valid_idx):
        vals = bucket.get(local_i, [])
        if vals:
            result[idx] = float(max(vals))
    return result


def compute_bartscore(samples: list, refs_lookup: dict) -> list:
    """
    BARTScore (bart-large-cnn), max over references.
    Returns List[float|None] — log-probability, typically negative.
    Supports CUDA / MPS / CPU.
    """
    import torch
    from bart_score import BARTScorer

    device = get_device()
    print(f'  BARTScore: loading facebook/bart-large-cnn on {device}')
    bart_scorer = BARTScorer(device=device, checkpoint='facebook/bart-large-cnn')
    if Path(BART_SCORE_PATH).exists():
        bart_scorer.load(path=BART_SCORE_PATH)
        print(f'  BARTScore: loaded fine-tuned weights from bart_score.pth')
    else:
        print(f'  BARTScore: bart_score.pth not found, using default checkpoint weights')

    # Flatten (hyp, ref) pairs — avoids padding issues with variable ref counts
    hyps_flat, refs_flat, owners = [], [], []
    for i, s in enumerate(tqdm(samples, desc='BARTScore (flatten)')):
        refs = s.get('refs')
        if refs is None:
            refs = get_refs(s['mr'], refs_lookup)
        for ref in refs:
            hyps_flat.append(s['output'])
            refs_flat.append(ref)
            owners.append(i)

    if not hyps_flat:
        return [None] * len(samples)

    print(f'  BARTScore: scoring {len(hyps_flat)} (hyp, ref) pairs')
    flat_scores = bart_scorer.score(hyps_flat, refs_flat, batch_size=BARTSCORE_BATCH)

    bucket = defaultdict(list)
    for i, sc in zip(owners, flat_scores):
        bucket[i].append(sc)

    result = [None] * len(samples)
    for i, s in enumerate(samples):
        refs = s.get('refs')
        if refs is None:
            refs = get_refs(s['mr'], refs_lookup)
        vals = bucket.get(i, [])
        if refs and vals:
            result[i] = float(max(vals))
    return result


def compute_bleurt(samples: list, refs_lookup: dict) -> list:
    """BLEURT (BLEURT-20), max over references. Returns List[float|None]."""
    if not Path(BLEURT_CHECKPOINT).exists():
        print(f'  BLEURT: checkpoint not found at {BLEURT_CHECKPOINT}, skipping')
        return [None] * len(samples)

    # On macOS: disable TF Metal to avoid SIGSEGV when PyTorch MPS is already
    # initialised (Metal resource conflict). On Linux/CUDA, leave GPU visible.
    import os, sys as _sys
    if _sys.platform == 'darwin':
        os.environ.setdefault('TF_METAL_ENABLED', '0')
        os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

    # baselines/bleurt/ is the git repo root; the Python package is baselines/bleurt/bleurt/.
    # Add the repo root to sys.path so "import bleurt" resolves to the inner package.
    _bleurt_repo = str(THIS_DIR / 'bleurt')
    if _bleurt_repo not in sys.path:
        sys.path.insert(0, _bleurt_repo)
    from bleurt import score as bleurt_score_mod
    print(f'  BLEURT: loading from {Path(BLEURT_CHECKPOINT).name}')
    scorer = bleurt_score_mod.BleurtScorer(BLEURT_CHECKPOINT)

    cands_flat, refs_flat, owners = [], [], []
    for i, s in enumerate(tqdm(samples, desc='BLEURT (flatten)')):
        refs = s.get('refs')
        if refs is None:
            refs = get_refs(s['mr'], refs_lookup)
        for ref in refs:
            cands_flat.append(s['output'])
            refs_flat.append(ref)
            owners.append(i)

    if not cands_flat:
        return [None] * len(samples)

    print(f'  BLEURT: scoring {len(cands_flat)} (hyp, ref) pairs')
    flat_scores = scorer.score(references=refs_flat, candidates=cands_flat)

    bucket = defaultdict(list)
    for i, sc in zip(owners, flat_scores):
        bucket[i].append(sc)

    result = [None] * len(samples)
    for i, s in enumerate(samples):
        refs = s.get('refs')
        if refs is None:
            refs = get_refs(s['mr'], refs_lookup)
        vals = bucket.get(i, [])
        if refs and vals:
            result[i] = float(max(vals))
    return result


# Main

def run():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for p in (CONVERTED, REFS_JSON):
        if not p.exists():
            raise FileNotFoundError(f'Required input not found: {p}')

    # Load data
    print('Loading converted.json ...')
    with open(CONVERTED, encoding='utf-8') as f:
        converted = json.load(f)
    samples = load_all_samples(converted)
    print(f'  {len(samples)} (mr_id, sys_name) pairs loaded')

    print('Loading refs_e2e19.json ...')
    with open(REFS_JSON, encoding='utf-8') as f:
        refs_lookup = json.load(f)
    for s in samples:
        s['refs'] = get_refs(s['mr'], refs_lookup)
    n_covered = sum(1 for s in samples if s['refs'])
    print(f'  {n_covered}/{len(samples)} samples covered by reference lookup')

    # Load cache
    cache: dict = {}
    if CACHE_FILE.exists():
        with open(CACHE_FILE, encoding='utf-8') as f:
            cache = json.load(f)
        print(f'Cache: {len(cache)} entries loaded from {CACHE_FILE.name}')

    def ck(s) -> str:
        return f'{s["mr_id"]}||{s["sys_name"]}'

    def has_metric(s, mkey: str) -> bool:
        entry = cache.get(ck(s), {})
        if mkey not in entry:
            return False

        v = entry[mkey]
        if v is None:
            # Only treat None as "done" when this sample truly has no reference.
            return not bool(s.get('refs'))

        try:
            return np.isfinite(float(v))
        except (TypeError, ValueError):
            return False

    def save_cache():
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False)
        print(f'  Cache saved: {len(cache)} entries')

    def store(todo_list: list, mkey: str, scores: list):
        for s, sc in zip(todo_list, scores):
            cache.setdefault(ck(s), {})[mkey] = sc

    # Stage 1: Inference

    if USE_BLEU:
        todo = [s for s in samples if not has_metric(s, 'bleu')]
        if todo:
            print(f'\nBLEU (sacrebleu sentence_bleu, {len(todo)} samples)')
            store(todo, 'bleu', compute_bleu(todo, refs_lookup))
            save_cache()
        else:
            print('BLEU: all cached')

    if USE_METEOR:
        todo = [s for s in samples if not has_metric(s, 'meteor')]
        if todo:
            print(f'\nMETEOR (nltk, {len(todo)} samples)')
            store(todo, 'meteor', compute_meteor(todo, refs_lookup))
            save_cache()
        else:
            print('METEOR: all cached')

    if USE_PARENT:
        todo = [s for s in samples if not has_metric(s, 'parent_f1')]
        if todo:
            print(f'\nPARENT (λ=0.5, pure Python, {len(todo)} samples)')
            store(todo, 'parent_f1', compute_parent(todo, refs_lookup))
            save_cache()
        else:
            print('PARENT: all cached')

    if USE_BERTSCORE:
        todo = [s for s in samples if not has_metric(s, 'bertscore_f1')]
        if todo:
            print(f'\nBERTScore (lang=en, {len(todo)} samples)')
            store(todo, 'bertscore_f1', compute_bertscore(todo, refs_lookup))
            save_cache()
        else:
            print('BERTScore: all cached')

    if USE_BARTSCORE:
        todo = [s for s in samples if not has_metric(s, 'bartscore')]
        if todo:
            print(f'\nBARTScore (bart-large-cnn, max over refs, {len(todo)} samples)')
            store(todo, 'bartscore', compute_bartscore(todo, refs_lookup))
            save_cache()
        else:
            print('BARTScore: all cached')

    if USE_BLEURT:
        todo = [s for s in samples if not has_metric(s, 'bleurt')]
        if todo:
            print(f'\nBLEURT (BLEURT-20, max over refs, {len(todo)} samples)')
            store(todo, 'bleurt', compute_bleurt(todo, refs_lookup))
            save_cache()
        else:
            print('BLEURT: all cached')

    # Stage 2: Worker filtering + correlations
    print('\nApplying worker filtering (drop_gt100_all + A+C+D+E+F) ...')
    add_human_quality(samples, converted)
    n_hq = sum(1 for s in samples if s.get('human_quality') is not None)
    print(f'  {n_hq}/{len(samples)} samples with valid filtered human quality')

    for s in samples:
        s.update(cache.get(ck(s), {}))

    METRICS = []
    if USE_BLEU:      METRICS.append(('bleu',        'BLEU'))
    if USE_METEOR:    METRICS.append(('meteor',       'METEOR'))
    if USE_PARENT:    METRICS.append(('parent_f1',    'PARENT-F1'))
    if USE_BERTSCORE: METRICS.append(('bertscore_f1', 'BERTScore-F1'))
    if USE_BARTSCORE: METRICS.append(('bartscore',    'BARTScore'))
    if USE_BLEURT:    METRICS.append(('bleurt',       'BLEURT'))

    W = 8
    header0 = (f"  {'Metric':<20} | {'n_valid':>7} |"
               f" {'Text-level':^{W * 3 + 2}} | {'System-level':^{W * 3 + 2}} |")
    header1 = (f"  {'':<20} | {'':>7} |"
               f" {'Pearson':>{W}} {'Spearman':>{W}} {'Kendall':>{W}} |"
               f" {'Pearson':>{W}} {'Spearman':>{W}} {'Kendall':>{W}} |")

    lines = [
        'Reference Baseline Eval — E2E19',
        'Metrics : BLEU | METEOR | PARENT | BERTScore | BARTScore | BLEURT',
        'Human   : filtered fine_score / 100  (drop_gt100_all + A+C+D+E+F)',
        '',
        header0, header1,
    ]

    def fmt(v):
        return f'{"nan":>{W}}' if np.isnan(v) else f'{v:>{W}.4f}'

    for mkey, mname in METRICS:
        text_lv, sys_lv, n_valid = compute_correlations(samples, mkey)
        lines.append(
            f"  {mname:<20} | {n_valid:>7} |"
            f" {fmt(text_lv[0])} {fmt(text_lv[1])} {fmt(text_lv[2])} |"
            f" {fmt(sys_lv[0])} {fmt(sys_lv[1])} {fmt(sys_lv[2])} |"
        )

    output = '\n'.join(lines) + '\n'
    print('\n' + output)

    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write(output)
    print(f'Log saved to {LOG_FILE}')


if __name__ == '__main__':
    run()
