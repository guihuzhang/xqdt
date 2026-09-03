#!/usr/bin/env python3
"""
baseline_eval_e2e19_add_pertri.py
----------------------------------
Extends baseline_scores_e2e19.json with per-triple score lists for NLI and
FactSpotter so that the coarse-label evaluator can apply "any triple fails"
thresholding instead of thresholding on the mean.
"""

import json
import re
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
E2E19_ROOT = SCRIPT_DIR.parent
CONVERTED = E2E19_ROOT / 'human_ratings/converted.json'
CACHE_FILE = SCRIPT_DIR / 'baseline_scores_e2e19.json'
FACTSPOTTER_ELECTRA_PATH = (PROJECT_ROOT / 'checkpoints/fact_spotter_electra.pt').resolve()
NLI_TEMPLATE_FILE = (SCRIPT_DIR / 'e2e_templates.json').resolve()

INFER_BATCH_SIZE = 16

NLI_E2E_REMAP = {
    'eat type': 'eatType',
    'family friendly': 'familyFriendly',
    'price range': 'priceRange',
}


def normalize_slot_name(slot_name: str) -> str:
    slot_name = slot_name.strip().replace('_', ' ')
    slot_name = re.sub(r'([a-z])([A-Z])', r'\1 \2', slot_name)
    slot_name = re.sub(r'([A-Z])([A-Z][a-z])', r'\1 \2', slot_name)
    slot_name = re.sub(r'\s+', ' ', slot_name)
    return slot_name.strip().lower()


def parse_e2e_mr_to_triples(mr_text: str):
    slot_pairs = [
        (m.group(1).strip(), m.group(2).strip())
        for m in re.finditer(r'\s*([^,\[]+?)\s*\[([^\]]*)\]\s*(?:,|$)', mr_text)
    ]
    subject = next(
        (val for slot, val in slot_pairs
         if normalize_slot_name(slot) == 'name' and val),
        'restaurant'
    )
    return [
        (subject, normalize_slot_name(slot), val)
        for slot, val in slot_pairs
        if val and normalize_slot_name(slot) != 'name'
    ]


def load_all_samples(converted: dict) -> list:
    samples = []
    for mr_id, mr_data in converted.items():
        mr_text = mr_data['mr']
        for sys_name, sys_data in mr_data.get('systems', {}).items():
            samples.append({
                'mr_id': mr_id,
                'sys_name': sys_name,
                'mr': mr_text,
                'output': sys_data.get('output', ''),
            })
    return samples


def iter_batches(items, batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def nli_load():
    from transformers import RobertaForSequenceClassification, RobertaTokenizer
    print('  Loading NLI (roberta-large-mnli) ...')
    tok = RobertaTokenizer.from_pretrained('roberta-large-mnli')
    mdl = RobertaForSequenceClassification.from_pretrained('roberta-large-mnli')
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        mdl.to('cuda')
    print(f'  ✅ NLI loaded ({"GPU" if use_gpu else "CPU"})')
    return tok, mdl, use_gpu


def nli_load_templates():
    try:
        with open(NLI_TEMPLATE_FILE, encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f'  ⚠️  e2e_templates.json not found at {NLI_TEMPLATE_FILE}; using backoff')
        return {}


def _nli_classify_batch(tok, mdl, use_gpu, pairs: list, sub_batch: int = 32):
    if not pairs:
        return np.empty((0, 3), dtype=float)
    all_probs = []
    packed_all = [f'{premise} </s></s> {hypothesis}' for premise, hypothesis in pairs]
    for start in range(0, len(packed_all), sub_batch):
        chunk = packed_all[start:start + sub_batch]
        inp = tok(chunk, return_tensors='pt', truncation=True, padding=True, max_length=512)
        if use_gpu:
            inp = {k: v.to('cuda') for k, v in inp.items()}
        with torch.no_grad():
            logits = mdl(**inp).logits
        if use_gpu:
            logits = logits.cpu()
        all_probs.append(torch.softmax(logits, dim=1).numpy())
    return np.concatenate(all_probs, axis=0)


def _nli_triple_to_sent(subj, pred_norm, obj, templates):
    key = NLI_E2E_REMAP.get(pred_norm, pred_norm)
    if key in templates:
        tpl = templates[key]
    else:
        tpl = f'The {pred_norm} of <subject> is <object>.'
    if isinstance(tpl, dict):
        tpl = tpl.get(obj, list(tpl.values())[0])
    if isinstance(tpl, list):
        tpl = tpl[0]
    obj_clean = re.sub(r'^["\'](.*)["\']$', r'\1', obj)
    return tpl.replace('<subject>', subj).replace('<object>', obj_clean).replace('_', ' ')


def factspotter_electra_load(model_path, base='google/electra-small-discriminator'):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    print(f'  Loading FactSpotter-ELECTRA ({Path(model_path).name}) ...')
    tok = AutoTokenizer.from_pretrained(base)
    mdl = AutoModelForSequenceClassification.from_pretrained(
        base, num_labels=2, ignore_mismatched_sizes=True)
    state = torch.load(model_path, map_location='cpu')
    mdl.load_state_dict(state, strict=False)
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        mdl.to('cuda')
    mdl.eval()
    print(f'  ✅ FactSpotter-ELECTRA loaded ({"GPU" if use_gpu else "CPU"})')
    return tok, mdl, use_gpu


def _unload_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def nli_rec_list_batch(tok, mdl, use_gpu, templates, texts, triples_batch):
    n = len(texts)
    rec_pairs = []
    rec_owner = []
    per_sample = [[] for _ in range(n)]

    for i, (text, triples) in enumerate(zip(texts, triples_batch)):
        if not triples:
            continue
        sents = [_nli_triple_to_sent(s, p, o, templates) for s, p, o in triples]
        for sent in sents:
            rec_pairs.append((text, sent))
            rec_owner.append(i)

    if rec_pairs:
        probs = _nli_classify_batch(tok, mdl, use_gpu, rec_pairs)
        for i, p in zip(rec_owner, probs[:, 2].tolist()):
            per_sample[i].append(float(p))

    return per_sample


def factspotter_list_batch(tok, mdl, use_gpu, texts, triples_batch, batch_size):
    try:
        from unidecode import unidecode
    except ImportError:
        unidecode = lambda x: x

    n = len(texts)
    flat_inputs = []
    flat_owner = []
    per_sample = [[] for _ in range(n)]

    for i, (text, triples) in enumerate(zip(texts, triples_batch)):
        if not triples:
            continue
        text_norm = unidecode(text.lower().strip())
        for subj, pred, obj in triples:
            flat_inputs.append(
                f'predicate: {unidecode(pred.lower().strip())}, '
                f'subject: {unidecode(subj.lower().strip())}, '
                f'object: {unidecode(obj.lower().strip())}, '
                f'sentence: {text_norm}'
            )
            flat_owner.append(i)

    if flat_inputs:
        for start in range(0, len(flat_inputs), batch_size):
            end = min(start + batch_size, len(flat_inputs))
            chunk = flat_inputs[start:end]
            owners = flat_owner[start:end]
            tkn = tok(chunk, truncation=True, padding=True, return_tensors='pt')
            if use_gpu:
                tkn = {k: v.to('cuda') for k, v in tkn.items()}
            with torch.no_grad():
                logits = mdl(**tkn).logits
            if use_gpu:
                logits = logits.cpu()
            probs = torch.softmax(logits, dim=1).numpy()[:, 1]
            for owner, prob in zip(owners, probs):
                per_sample[owner].append(float(prob))

    return per_sample


def run():
    if not CONVERTED.exists():
        raise FileNotFoundError(f'Human ratings file not found: {CONVERTED}')
    if not CACHE_FILE.exists():
        raise FileNotFoundError(f'Baseline cache not found: {CACHE_FILE}')

    print('Loading converted.json ...')
    with open(CONVERTED, encoding='utf-8') as f:
        converted = json.load(f)
    samples = load_all_samples(converted)
    print(f'  {len(samples)} (mr_id, sys_name) pairs')

    print(f'Loading cache from {CACHE_FILE.name} ...')
    with open(CACHE_FILE, encoding='utf-8') as f:
        cache = json.load(f)
    print(f'  {len(cache)} entries in cache')

    def cache_key(sample):
        return f'{sample["mr_id"]}||{sample["sys_name"]}'

    def save_cache():
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False)
        print(f'  → Cache saved: {CACHE_FILE.name}')

    nli_todo = [s for s in samples if 'nli_rec_list' not in cache.get(cache_key(s), {})]
    print(f'\n── NLI per-triple recall: {len(nli_todo)} samples to process ─────────────')
    if nli_todo:
        tok, mdl, use_gpu = nli_load()
        templates = nli_load_templates()
        failed = 0
        total = (len(nli_todo) + INFER_BATCH_SIZE - 1) // INFER_BATCH_SIZE

        for batch in tqdm(iter_batches(nli_todo, INFER_BATCH_SIZE), total=total, desc='NLI per-triple'):
            texts = [s['output'] for s in batch]
            triples_batch = [parse_e2e_mr_to_triples(s['mr']) for s in batch]
            try:
                rec_lists = nli_rec_list_batch(tok, mdl, use_gpu, templates, texts, triples_batch)
                for sample, rec_list in zip(batch, rec_lists):
                    key = cache_key(sample)
                    cache.setdefault(key, {})
                    cache[key]['nli_rec_list'] = rec_list
            except Exception as exc:
                failed += len(batch)
                print(f'  batch error: {exc}')

        del tok, mdl
        _unload_gpu()
        print(f'  NLI per-triple done. failed batches approx: {failed}')
        save_cache()
    else:
        print('  Already complete, skipping.')

    fs_todo = [s for s in samples if 'factspotter_list' not in cache.get(cache_key(s), {})]
    print(f'\n── FactSpotter per-triple: {len(fs_todo)} samples to process ──')
    if fs_todo:
        if not FACTSPOTTER_ELECTRA_PATH.exists():
            print(f'  ⚠️  Model not found: {FACTSPOTTER_ELECTRA_PATH}; skipping')
        else:
            tok, mdl, use_gpu = factspotter_electra_load(str(FACTSPOTTER_ELECTRA_PATH))
            failed = 0
            total = (len(fs_todo) + INFER_BATCH_SIZE - 1) // INFER_BATCH_SIZE

            for batch in tqdm(iter_batches(fs_todo, INFER_BATCH_SIZE), total=total, desc='FactSpotter per-triple'):
                texts = [s['output'] for s in batch]
                triples_batch = [parse_e2e_mr_to_triples(s['mr']) for s in batch]
                try:
                    fs_lists = factspotter_list_batch(tok, mdl, use_gpu, texts, triples_batch, INFER_BATCH_SIZE)
                    for sample, fs_list in zip(batch, fs_lists):
                        key = cache_key(sample)
                        cache.setdefault(key, {})
                        cache[key]['factspotter_list'] = fs_list
                except Exception as exc:
                    failed += len(batch)
                    print(f'  batch error: {exc}')

            del tok, mdl
            _unload_gpu()
            print(f'  FactSpotter per-triple done. failed batches approx: {failed}')
            save_cache()
    else:
        print('  Already complete, skipping.')

    print('\nDone.')


if __name__ == '__main__':
    run()
