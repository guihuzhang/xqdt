#!/usr/bin/env python3
"""
Unified 4L-RP-Human Baseline Evaluation
- MonoLR (William's method)
- NLI (Dusek's method)
- legacy FactSpotter-ELECTRA - RECALL ONLY (paper FS row)
- optional FactSpotter-DeBERTa variants - RECALL ONLY (not the paper FS row)
Compare enabled baselines with bootstrap correlation analysis
"""

import argparse
import json
import torch
import re
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr, kendalltau
from pathlib import Path
from datetime import datetime

# MonoLR imports
from sentence_transformers import CrossEncoder

# NLI imports
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# FactSpotter imports
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.utils.rnn import pad_sequence
from unidecode import unidecode

# Random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

SCRIPT_DIR = Path(__file__).resolve().parent
CORRELATION_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT_JSON = CORRELATION_ROOT / "by_language" / "english.json"
DEFAULT_TEMPLATE_FILE = SCRIPT_DIR / "webnlg_templates.json"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results"
DEFAULT_FACTSPOTTER_ELECTRA_PATH = CORRELATION_ROOT.parents[1] / "checkpoints" / "fact_spotter_electra.pt"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_json(path: Path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=json_default)


def write_summary_log(output_dir: Path, config: dict, correlations: dict):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"plm_baselines_evaluation_{timestamp}.log"
    lines = [
        "4L-RP-Human baseline evaluation",
        f"input_json: {config['input_json']}",
        f"template_file: {config['template_file']}",
        f"factspotter_electra_path: {config['factspotter_electra_path']}",
        f"output_dir: {output_dir}",
        "",
    ]

    for key, corr in correlations.items():
        if corr is None:
            lines.append(f"{key}: skipped")
            continue

        if "spearman_prec" in corr:
            lines.extend([
                key,
                f"  Pearson:  P={corr['pearson_prec']:.4f} R={corr['pearson_rec']:.4f} F1={corr['pearson_f1']:.4f}",
                f"  Spearman: P={corr['spearman_prec']:.4f} R={corr['spearman_rec']:.4f} F1={corr['spearman_f1']:.4f}",
                f"  Kendall:  P={corr['kendall_prec']:.4f} R={corr['kendall_rec']:.4f} F1={corr['kendall_f1']:.4f}",
                "",
            ])
        else:
            lines.extend([
                key,
                f"  Pearson:  R={corr['pearson']:.4f}",
                f"  Spearman: R={corr['spearman']:.4f}",
                f"  Kendall:  R={corr['kendall']:.4f}",
                f"  RMSE:     R={corr['rmse']:.4f}",
                "",
            ])

    log_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return log_path


# ============================================================================
# MONOLR BASELINE (William's Paper)
# ============================================================================

def monolr_william_load_model():
    """Load MonoLR model with adapter (William's method)"""
    print("Loading MonoLR model (William's method)...")
    model = CrossEncoder('MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7')
    model.config.num_labels = 1
    model.model.load_adapter('WilliamSotoM/MonoLR_eng_Latn_PR')
    print("✅ MonoLR (William) loaded!")
    return model


def monolr_william_format_graph(graph_str: str) -> str:
    """Convert graph to MonoLR format: [S]A[P]B[O]C[T]..."""
    triples = graph_str.split('<br>')
    formatted_parts = []

    for triple in triples:
        parts = [p.strip() for p in triple.split('|')]
        if len(parts) == 3:
            s, p, o = [x.replace(' ', '_') for x in parts]
            formatted_parts.append(f"[S]{s}[P]{p}[O]{o}")

    return '[T]'.join(formatted_parts)


def monolr_william_compute_scores(model, text: str, graph: str):
    """Compute precision, recall, F1 using MonoLR"""
    # Precision: (graph, text)
    precision_logit = model.predict([(graph, text)])[0]
    precision = float(torch.sigmoid(torch.tensor(precision_logit)).item())

    # Recall: (text, graph)
    recall_logit = model.predict([(text, graph)])[0]
    recall = float(torch.sigmoid(torch.tensor(recall_logit)).item())

    # F1
    if precision + recall > 0:
        f1 = (2 * precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    return precision, recall, f1


# ============================================================================
# NLI-BASED BASELINE (Dusek's Paper)
# ============================================================================

TEMPLATE_REMAP = {
    'eat_type': 'eatType',
    'rating': 'customer rating',
    'family_friendly': 'familyFriendly',
    'price_range': 'priceRange',
}


def nli_dusek_load_model():
    """Load RoBERTa-large-MNLI model (Dusek's method)"""
    print("Loading NLI model (Dusek's method)...")

    tokenizer = RobertaTokenizer.from_pretrained('roberta-large-mnli')
    model = RobertaForSequenceClassification.from_pretrained('roberta-large-mnli')

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model.to('cuda')
        print("✅ NLI (Dusek) loaded on GPU!")
    else:
        print("✅ NLI (Dusek) loaded on CPU!")

    return tokenizer, model, use_gpu


def nli_dusek_load_templates(template_file: str = None):
    """Load WebNLG templates (Dusek's method)"""
    if not template_file:
        return {}

    try:
        with open(template_file, 'r', encoding='utf-8') as f:
            templates = json.load(f)
        print(f"✅ Loaded {len(templates)} templates (Dusek)")
        return templates
    except FileNotFoundError:
        print(f"⚠️  Template file not found, using backoff only")
        return {}


def nli_dusek_parse_triple(triple_str: str):
    """Parse triple string: 'S | P | O'"""
    parts = [p.strip() for p in triple_str.split('|')]
    if len(parts) == 3:
        return {'subject': parts[0], 'predicate': parts[1], 'object': parts[2]}
    return None


def nli_dusek_triple_to_template(triple: dict, templates: dict):
    """Convert triple to template sentence"""
    predicate = triple['predicate']
    subject = triple['subject']
    obj = triple['object']

    # Find template
    if predicate not in templates:
        if predicate in TEMPLATE_REMAP and TEMPLATE_REMAP[predicate] in templates:
            template = templates[TEMPLATE_REMAP[predicate]]
        else:
            # Backoff template
            template = f"The {predicate} of <subject> is <object>."
            template = re.sub('([a-z])([A-Z])', r"\1 \2", template)
    else:
        template = templates[predicate]

    # Handle different formats
    if isinstance(template, dict):
        template = template.get(obj, list(template.values())[0])
    if isinstance(template, list):
        template = template[0]

    # Fill in subject and object
    template = template.replace('<subject>', subject)
    obj_str = re.sub('^["\'](.*)["\']$', r'\1', obj)
    template = template.replace('<object>', obj_str)
    template = template.replace('_', ' ')

    return template


def nli_dusek_roberta_classify(tokenizer, model, use_gpu, premise, hypothesis):
    """Classify entailment using RoBERTa-MNLI"""
    inputs = tokenizer(f"{premise} </s></s> {hypothesis}", return_tensors="pt")

    if use_gpu:
        inputs = inputs.to('cuda')

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    if use_gpu:
        logits = logits.cpu()

    probs = torch.nn.Softmax(dim=1)(logits).numpy()[0]
    return probs  # [contradiction, neutral, entailment]


def nli_dusek_compute_scores(tokenizer, model, use_gpu, text: str, triples: list, templates: dict):
    """Compute NLI-based scores using continuous entailment scores"""
    # Convert triples to templates
    template_sents = []

    for triple_str in triples:
        triple_dict = nli_dusek_parse_triple(triple_str)
        if triple_dict:
            template = nli_dusek_triple_to_template(triple_dict, templates)
            template_sents.append(template)

    if not template_sents:
        return 0.0, 0.0, 0.0

    # Precision: Global MR → text
    all_templates = ' '.join(template_sents)
    probs_mr2text = nli_dusek_roberta_classify(tokenizer, model, use_gpu, all_templates, text)
    precision = float(probs_mr2text[2])

    # Recall: Average text → each triple
    triple_scores = []
    for template in template_sents:
        probs_text2triple = nli_dusek_roberta_classify(tokenizer, model, use_gpu, text, template)
        triple_scores.append(float(probs_text2triple[2]))

    recall = float(np.mean(triple_scores)) if triple_scores else 0.0

    # F1
    if precision + recall > 0:
        f1 = (2 * precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    return precision, recall, f1


# ============================================================================
# FACTSPOTTER-ELECTRA BASELINE (legacy paper model) - RECALL ONLY
# ============================================================================

def factspotter_electra_load_model(model_path: str, base_model_name: str = "google/electra-small-discriminator"):
    """
    Load FactSpotter-ELECTRA model from checkpoint

    Args:
        model_path: Path to the .pt checkpoint file
        base_model_name: Base model architecture (default: google/electra-small-discriminator)
    """
    print("Loading FactSpotter-ELECTRA model...")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=2,
        ignore_mismatched_sizes=True
    )

    # Load trained weights
    state_dict = torch.load(model_path, map_location='cpu')
    # checkpoint_keys = set(state_dict.keys())
    # model_keys = set(model.state_dict().keys())
    # only_in_checkpoint = checkpoint_keys - model_keys
    # if only_in_checkpoint:
    #     print(f"\n⚠️  Keys ONLY in checkpoint ({len(only_in_checkpoint)}):")
    #     for key in sorted(only_in_checkpoint):
    #         print(f"   - {key}")
    #
    # # Keys only in model
    # only_in_model = model_keys - checkpoint_keys
    # if only_in_model:
    #     print(f"\n⚠️  Keys ONLY in model ({len(only_in_model)}):")
    #     for key in sorted(only_in_model):
    #         print(f"   - {key}")
    #
    # # Common keys
    # common_keys = checkpoint_keys & model_keys
    # print(f"\n✅ Common keys: {len(common_keys)}")

    model.load_state_dict(state_dict, strict=False)

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model.to('cuda')
        print("✅ FactSpotter-ELECTRA loaded on GPU!")
    else:
        print("✅ FactSpotter-ELECTRA loaded on CPU!")

    model.eval()
    return tokenizer, model, use_gpu


def factspotter_electra_format_input(text: str, subject: str, predicate: str, obj: str) -> str:
    """
    Format input for FactSpotter-ELECTRA model
    Format: "predicate: P, subject: S, object: O, sentence: TEXT"
    """
    return f"predicate: {predicate}, subject: {subject}, object: {obj}, sentence: {text}"


def factspotter_electra_classify_triple(tokenizer, model, use_gpu, text: str, subject: str, predicate: str, obj: str):
    """
    Classify if a triple is supported by the text
    Returns probability that the triple is correct (label=1)
    """
    # Format input
    input_str = factspotter_electra_format_input(text, subject, predicate, obj)

    # Tokenize
    tokenized = tokenizer([input_str], truncation=True, return_tensors="pt")

    if use_gpu:
        tokenized = {k: v.to('cuda') for k, v in tokenized.items()}

    # Inference
    with torch.no_grad():
        outputs = model(**tokenized)
        logits = outputs.logits

        if use_gpu:
            logits = logits.cpu()

        # Get probability of label=1 (triple is correct)
        probs = torch.nn.Softmax(dim=1)(logits).numpy()[0]
        return float(probs[1])  # Probability of class 1


def factspotter_electra_compute_recall(tokenizer, model, use_gpu, text: str, triples: list):
    """
    Compute RECALL ONLY using FactSpotter-ELECTRA
    Recall: Average score across all triples (text → triple)
    """
    triple_scores = []

    for triple_str in triples:
        parts = [p.strip() for p in triple_str.split('|')]
        if len(parts) == 3:
            subject, predicate, obj = parts

            # Normalize text (following training preprocessing)
            text_normalized = unidecode(text.lower().strip())
            subject_normalized = unidecode(subject.lower().strip())
            predicate_normalized = unidecode(predicate.lower().strip())
            obj_normalized = unidecode(obj.lower().strip())

            # Get FactSpotter score
            score = factspotter_electra_classify_triple(
                tokenizer, model, use_gpu,
                text_normalized, subject_normalized, predicate_normalized, obj_normalized
            )
            triple_scores.append(score)

    # Recall is average score across all triples
    recall = float(np.mean(triple_scores)) if triple_scores else 0.0

    return recall


# ============================================================================
# FACTSPOTTER-DEBERTA BASELINE (Inria-CEDAR) - RECALL ONLY
# ============================================================================

def factspotter_deberta_load_model(model_name: str):
    """
    Load FactSpotter-DeBERTa model from Hugging Face

    Args:
        model_name: "Inria-CEDAR/FactSpotter-DeBERTaV3-Base" or "Inria-CEDAR/FactSpotter-DeBERTaV3-Large"
    """
    print(f"Loading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model.to('cuda')
        print(f"✅ {model_name.split('/')[-1]} loaded on GPU!")
    else:
        print(f"✅ {model_name.split('/')[-1]} loaded on CPU!")

    model.eval()
    return tokenizer, model, use_gpu


def factspotter_deberta_sentence_cls_score(input_strings, model, tokenizer, use_gpu):
    """
    Compute classification scores for FactSpotter-DeBERTa
    Input: List of (premise, hypothesis) tuples
    Output: Softmax scores [entailment, neutral, contradiction]
    """
    tokenized_cls_input = tokenizer(
        input_strings,
        truncation=True,
        padding=True,
        return_token_type_ids=True,
        return_tensors='pt'
    )

    if use_gpu:
        input_ids = tokenized_cls_input['input_ids'].to('cuda')
        token_type_ids = tokenized_cls_input['token_type_ids'].to('cuda')
        attention_mask = tokenized_cls_input['attention_mask'].to('cuda')
    else:
        input_ids = tokenized_cls_input['input_ids']
        token_type_ids = tokenized_cls_input['token_type_ids']
        attention_mask = tokenized_cls_input['attention_mask']

    with torch.no_grad():
        prev_cls_output = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        softmax_cls_output = torch.softmax(prev_cls_output.logits, dim=1)

    if use_gpu:
        softmax_cls_output = softmax_cls_output.cpu()

    return softmax_cls_output.numpy()


def factspotter_deberta_compute_recall(tokenizer, model, use_gpu, text: str, triples: list):
    """
    Compute RECALL ONLY using FactSpotter-DeBERTa
    Recall: Average entailment score across all triples (text → triple)

    Input format: (text, "subject, predicate, object")
    Output: Dimensions: 0-entailment, 1-neutral, 2-contradiction
    """
    # Prepare input pairs
    cls_texts = []

    for triple_str in triples:
        parts = [p.strip() for p in triple_str.split('|')]
        if len(parts) == 3:
            subject, predicate, obj = parts
            # Format: (premise, hypothesis) where hypothesis is "subject, predicate, object"
            hypothesis = f"{subject}, {predicate}, {obj}"
            cls_texts.append((text, hypothesis))

    if not cls_texts:
        return 0.0

    # Get scores
    cls_scores = factspotter_deberta_sentence_cls_score(cls_texts, model, tokenizer, use_gpu)

    # Extract entailment scores (dimension 0)
    entailment_scores = cls_scores[:, 0]

    # Recall is average entailment score
    recall = float(np.mean(entailment_scores))

    return recall


# ============================================================================
# INFERENCE
# ============================================================================

def run_inference(input_json: str, template_file: str = None,
                  factspotter_electra_path: str = None,
                  factspotter_electra_base: str = "google/electra-small-discriminator",
                  use_factspotter_deberta_small: bool = False,  # ✅ 添加 Small
                  use_factspotter_deberta_base: bool = False,
                  use_factspotter_deberta_large: bool = False):
    """Run all baselines on the input data"""
    print("\n" + "=" * 70)
    print("RUNNING BASELINES")
    print("=" * 70)

    # Load data
    print(f"\n📂 Loading: {input_json}")
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"📊 Total samples: {len(data)}")

    # Load models
    print("\n" + "─" * 70)
    print("Loading Models...")
    print("─" * 70)

    # MonoLR (William's method)
    monolr_model = monolr_william_load_model()

    # NLI (Dusek's method)
    nli_tokenizer, nli_model, nli_use_gpu = nli_dusek_load_model()
    nli_templates = nli_dusek_load_templates(template_file)

    # legacy FactSpotter-ELECTRA (paper FS row) - RECALL ONLY
    fs_electra_tokenizer, fs_electra_model, fs_electra_use_gpu = None, None, None
    if factspotter_electra_path:
        fs_electra_tokenizer, fs_electra_model, fs_electra_use_gpu = factspotter_electra_load_model(
            factspotter_electra_path, factspotter_electra_base
        )
    else:
        print("⚠️  FactSpotter-ELECTRA model path not provided, skipping")

    # ✅ FactSpotter-DeBERTa-Small (Inria-CEDAR) - RECALL ONLY
    fs_deberta_small_tokenizer, fs_deberta_small_model, fs_deberta_small_use_gpu = None, None, None
    if use_factspotter_deberta_small:
        fs_deberta_small_tokenizer, fs_deberta_small_model, fs_deberta_small_use_gpu = factspotter_deberta_load_model(
            "Inria-CEDAR/FactSpotter-DeBERTaV3-Small"
        )
    else:
        print("⚠️  FactSpotter-DeBERTa-Small not enabled, skipping")

    # FactSpotter-DeBERTa-Base (Inria-CEDAR) - RECALL ONLY
    fs_deberta_base_tokenizer, fs_deberta_base_model, fs_deberta_base_use_gpu = None, None, None
    if use_factspotter_deberta_base:
        fs_deberta_base_tokenizer, fs_deberta_base_model, fs_deberta_base_use_gpu = factspotter_deberta_load_model(
            "Inria-CEDAR/FactSpotter-DeBERTaV3-Base"
        )
    else:
        print("⚠️  FactSpotter-DeBERTa-Base not enabled, skipping")

    # FactSpotter-DeBERTa-Large (Inria-CEDAR) - RECALL ONLY
    fs_deberta_large_tokenizer, fs_deberta_large_model, fs_deberta_large_use_gpu = None, None, None
    if use_factspotter_deberta_large:
        fs_deberta_large_tokenizer, fs_deberta_large_model, fs_deberta_large_use_gpu = factspotter_deberta_load_model(
            "Inria-CEDAR/FactSpotter-DeBERTaV3-Large"
        )
    else:
        print("⚠️  FactSpotter-DeBERTa-Large not enabled, skipping")

    # Run inference
    results_monolr = []
    results_nli = []
    results_fs_electra = []
    results_fs_deberta_small = []  # ✅ 添加 Small
    results_fs_deberta_base = []
    results_fs_deberta_large = []

    print("\n" + "─" * 70)
    print("Running Inference...")
    print("─" * 70)

    for item in tqdm(data, desc="Processing"):
        text = item['text']
        graph_raw = item['graph']
        triples = [t.strip() for t in graph_raw.split('<br>') if t.strip()]

        try:
            # MonoLR (William)
            graph_monolr = monolr_william_format_graph(graph_raw)
            prec_m, rec_m, f1_m = monolr_william_compute_scores(monolr_model, text, graph_monolr)

            results_monolr.append({
                'id': item['id'],
                'question_id': item['question_id'],
                'text': text,
                'graph': graph_raw,
                'precision': prec_m,
                'recall': rec_m,
                'f1': f1_m,
                'human_precision_list': item.get('precision_list', []),
                'human_recall_list': item.get('recall_list', []),
            })

            # NLI (Dusek)
            prec_n, rec_n, f1_n = nli_dusek_compute_scores(
                nli_tokenizer, nli_model, nli_use_gpu, text, triples, nli_templates
            )

            results_nli.append({
                'id': item['id'],
                'question_id': item['question_id'],
                'text': text,
                'graph': graph_raw,
                'precision': prec_n,
                'recall': rec_n,
                'f1': f1_n,
                'human_precision_list': item.get('precision_list', []),
                'human_recall_list': item.get('recall_list', []),
            })

            # FactSpotter-ELECTRA - RECALL ONLY
            if fs_electra_model:
                rec_fse = factspotter_electra_compute_recall(
                    fs_electra_tokenizer, fs_electra_model, fs_electra_use_gpu, text, triples
                )

                results_fs_electra.append({
                    'id': item['id'],
                    'question_id': item['question_id'],
                    'text': text,
                    'graph': graph_raw,
                    'recall': rec_fse,
                    'human_recall_list': item.get('recall_list', []),
                })

            # ✅ FactSpotter-DeBERTa-Small - RECALL ONLY
            if fs_deberta_small_model:
                rec_fsds = factspotter_deberta_compute_recall(
                    fs_deberta_small_tokenizer, fs_deberta_small_model, fs_deberta_small_use_gpu, text, triples
                )

                results_fs_deberta_small.append({
                    'id': item['id'],
                    'question_id': item['question_id'],
                    'text': text,
                    'graph': graph_raw,
                    'recall': rec_fsds,
                    'human_recall_list': item.get('recall_list', []),
                })

            # FactSpotter-DeBERTa-Base - RECALL ONLY
            if fs_deberta_base_model:
                rec_fsdb = factspotter_deberta_compute_recall(
                    fs_deberta_base_tokenizer, fs_deberta_base_model, fs_deberta_base_use_gpu, text, triples
                )

                results_fs_deberta_base.append({
                    'id': item['id'],
                    'question_id': item['question_id'],
                    'text': text,
                    'graph': graph_raw,
                    'recall': rec_fsdb,
                    'human_recall_list': item.get('recall_list', []),
                })

            # FactSpotter-DeBERTa-Large - RECALL ONLY
            if fs_deberta_large_model:
                rec_fsdl = factspotter_deberta_compute_recall(
                    fs_deberta_large_tokenizer, fs_deberta_large_model, fs_deberta_large_use_gpu, text, triples
                )

                results_fs_deberta_large.append({
                    'id': item['id'],
                    'question_id': item['question_id'],
                    'text': text,
                    'graph': graph_raw,
                    'recall': rec_fsdl,
                    'human_recall_list': item.get('recall_list', []),
                })

        except Exception as e:
            print(f"\n⚠️  Error on {item['id']}: {e}")
            import traceback
            traceback.print_exc()
            continue

    return results_monolr, results_nli, results_fs_electra, results_fs_deberta_small, results_fs_deberta_base, results_fs_deberta_large


# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================

def compute_correlations_recall_only(results: list, model_name: str, n_bootstrap: int = 1000, confidence: float = 0.95):
    """Compute correlations with bootstrap (RECALL ONLY)"""
    print(f"\n{'=' * 70}")
    print(f"Computing Correlations: {model_name}")
    print('=' * 70)

    model_recall = []
    human_recall_avg = []

    for item in results:
        if len(item.get('human_recall_list', [])) == 0:
            continue

        model_recall.append(item['recall'])
        h_rec = np.mean(item['human_recall_list'])
        human_recall_avg.append(h_rec)

    if len(model_recall) == 0:
        print("⚠️  No valid samples!")
        return None

    model_recall = np.array(model_recall)
    human_recall_avg = np.array(human_recall_avg)

    pearson_rec = pearsonr(model_recall, human_recall_avg)
    spearman_rec = spearmanr(model_recall, human_recall_avg)
    kendall_rec = kendalltau(model_recall, human_recall_avg)
    rmse_rec = np.sqrt(np.mean((model_recall - human_recall_avg) ** 2))

    print("📊 WITHOUT Bootstrap - RECALL ONLY")
    print("📈 Pearson:")
    print(f"   Recall:    {pearson_rec[0]:.4f} (p={pearson_rec[1]:.4f})")
    print("📈 Spearman:")
    print(f"   Recall:    {spearman_rec[0]:.4f} (p={spearman_rec[1]:.4f})")
    print("📈 Kendall:")
    print(f"   Recall:    {kendall_rec[0]:.4f} (p={kendall_rec[1]:.4f})")
    print("📉 RMSE:")
    print(f"   Recall:    {rmse_rec:.4f}")

    n_samples = len(model_recall)
    alpha = 1 - confidence

    bootstrap_pearson_rec = []
    bootstrap_spearman_rec = []
    bootstrap_kendall_rec = []
    bootstrap_rmse_rec = []

    for _ in tqdm(range(n_bootstrap), desc="Bootstrap", leave=False):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        mr = model_recall[indices]
        hr = human_recall_avg[indices]
        bootstrap_pearson_rec.append(pearsonr(mr, hr)[0])
        bootstrap_spearman_rec.append(spearmanr(mr, hr)[0])
        bootstrap_kendall_rec.append(kendalltau(mr, hr)[0])
        bootstrap_rmse_rec.append(np.sqrt(np.mean((mr - hr) ** 2)))

    def get_ci(values, alpha):
        return np.percentile(values, alpha / 2 * 100), np.percentile(values, (1 - alpha / 2) * 100)

    pearson_rec_ci = get_ci(bootstrap_pearson_rec, alpha)
    spearman_rec_ci = get_ci(bootstrap_spearman_rec, alpha)
    kendall_rec_ci = get_ci(bootstrap_kendall_rec, alpha)
    rmse_rec_ci = get_ci(bootstrap_rmse_rec, alpha)

    print(f"📊 WITH Bootstrap (n={n_bootstrap}, {int(confidence * 100)}% CI) - RECALL ONLY")
    print("📈 Pearson:")
    print(f"   Recall:    {np.mean(bootstrap_pearson_rec):.4f} [{pearson_rec_ci[0]:.4f}, {pearson_rec_ci[1]:.4f}]")
    print("📈 Spearman:")
    print(f"   Recall:    {np.mean(bootstrap_spearman_rec):.4f} [{spearman_rec_ci[0]:.4f}, {spearman_rec_ci[1]:.4f}]")
    print("📈 Kendall:")
    print(f"   Recall:    {np.mean(bootstrap_kendall_rec):.4f} [{kendall_rec_ci[0]:.4f}, {kendall_rec_ci[1]:.4f}]")
    print("📉 RMSE:")
    print(f"   Recall:    {np.mean(bootstrap_rmse_rec):.4f} [{rmse_rec_ci[0]:.4f}, {rmse_rec_ci[1]:.4f}]")

    return {
        'model': model_name,
        'pearson': pearson_rec[0],
        'spearman': spearman_rec[0],
        'kendall': kendall_rec[0],
        'rmse': rmse_rec
    }


def compute_correlations_full(results: list, model_name: str, n_bootstrap: int = 1000, confidence: float = 0.95):
    """Compute correlations with bootstrap (for MonoLR and NLI - full metrics)"""
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

    print("📊 WITHOUT Bootstrap")
    print("📈 Pearson:")
    print(f"   Precision: {pearson_prec[0]:.4f} (p={pearson_prec[1]:.4f})")
    print(f"   Recall:    {pearson_rec[0]:.4f} (p={pearson_rec[1]:.4f})")
    print(f"   F1:        {pearson_f1[0]:.4f} (p={pearson_f1[1]:.4f})")
    print("📈 Spearman:")
    print(f"   Precision: {spearman_prec[0]:.4f} (p={spearman_prec[1]:.4f})")
    print(f"   Recall:    {spearman_rec[0]:.4f} (p={spearman_rec[1]:.4f})")
    print(f"   F1:        {spearman_f1[0]:.4f} (p={spearman_f1[1]:.4f})")
    print("📈 Kendall:")
    print(f"   Precision: {kendall_prec[0]:.4f} (p={kendall_prec[1]:.4f})")
    print(f"   Recall:    {kendall_rec[0]:.4f} (p={kendall_rec[1]:.4f})")
    print(f"   F1:        {kendall_f1[0]:.4f} (p={kendall_f1[1]:.4f})")
    print("📉 RMSE:")
    print(f"   Precision: {rmse_prec:.4f}")
    print(f"   Recall:    {rmse_rec:.4f}")
    print(f"   F1:        {rmse_f1:.4f}")

    n_samples = len(model_precision)
    alpha = 1 - confidence

    bootstrap_pearson_prec, bootstrap_pearson_rec, bootstrap_pearson_f1 = [], [], []
    bootstrap_spearman_prec, bootstrap_spearman_rec, bootstrap_spearman_f1 = [], [], []
    bootstrap_kendall_prec, bootstrap_kendall_rec, bootstrap_kendall_f1 = [], [], []
    bootstrap_rmse_prec, bootstrap_rmse_rec, bootstrap_rmse_f1 = [], [], []

    for _ in tqdm(range(n_bootstrap), desc="Bootstrap", leave=False):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)

        mp, mr, mf = model_precision[indices], model_recall[indices], model_f1[indices]
        hp, hr, hf = human_precision_avg[indices], human_recall_avg[indices], human_f1_avg[indices]

        bootstrap_pearson_prec.append(pearsonr(mp, hp)[0])
        bootstrap_pearson_rec.append(pearsonr(mr, hr)[0])
        bootstrap_pearson_f1.append(pearsonr(mf, hf)[0])

        bootstrap_spearman_prec.append(spearmanr(mp, hp)[0])
        bootstrap_spearman_rec.append(spearmanr(mr, hr)[0])
        bootstrap_spearman_f1.append(spearmanr(mf, hf)[0])

        bootstrap_kendall_prec.append(kendalltau(mp, hp)[0])
        bootstrap_kendall_rec.append(kendalltau(mr, hr)[0])
        bootstrap_kendall_f1.append(kendalltau(mf, hf)[0])

        bootstrap_rmse_prec.append(np.sqrt(np.mean((mp - hp) ** 2)))
        bootstrap_rmse_rec.append(np.sqrt(np.mean((mr - hr) ** 2)))
        bootstrap_rmse_f1.append(np.sqrt(np.mean((mf - hf) ** 2)))

    def get_ci(values, alpha):
        return np.percentile(values, alpha / 2 * 100), np.percentile(values, (1 - alpha / 2) * 100)

    pearson_prec_ci = get_ci(bootstrap_pearson_prec, alpha)
    pearson_rec_ci = get_ci(bootstrap_pearson_rec, alpha)
    pearson_f1_ci = get_ci(bootstrap_pearson_f1, alpha)

    spearman_prec_ci = get_ci(bootstrap_spearman_prec, alpha)
    spearman_rec_ci = get_ci(bootstrap_spearman_rec, alpha)
    spearman_f1_ci = get_ci(bootstrap_spearman_f1, alpha)

    kendall_prec_ci = get_ci(bootstrap_kendall_prec, alpha)
    kendall_rec_ci = get_ci(bootstrap_kendall_rec, alpha)
    kendall_f1_ci = get_ci(bootstrap_kendall_f1, alpha)

    rmse_prec_ci = get_ci(bootstrap_rmse_prec, alpha)
    rmse_rec_ci = get_ci(bootstrap_rmse_rec, alpha)
    rmse_f1_ci = get_ci(bootstrap_rmse_f1, alpha)

    print(f"📊 WITH Bootstrap (n={n_bootstrap}, {int(confidence * 100)}% CI)")
    print("📈 Pearson:")
    print(f"   Precision: {np.mean(bootstrap_pearson_prec):.4f} [{pearson_prec_ci[0]:.4f}, {pearson_prec_ci[1]:.4f}]")
    print(f"   Recall:    {np.mean(bootstrap_pearson_rec):.4f} [{pearson_rec_ci[0]:.4f}, {pearson_rec_ci[1]:.4f}]")
    print(f"   F1:        {np.mean(bootstrap_pearson_f1):.4f} [{pearson_f1_ci[0]:.4f}, {pearson_f1_ci[1]:.4f}]")
    print("📈 Spearman:")
    print(
        f"   Precision: {np.mean(bootstrap_spearman_prec):.4f} [{spearman_prec_ci[0]:.4f}, {spearman_prec_ci[1]:.4f}]")
    print(f"   Recall:    {np.mean(bootstrap_spearman_rec):.4f} [{spearman_rec_ci[0]:.4f}, {spearman_rec_ci[1]:.4f}]")
    print(f"   F1:        {np.mean(bootstrap_spearman_f1):.4f} [{spearman_f1_ci[0]:.4f}, {spearman_f1_ci[1]:.4f}]")
    print("📈 Kendall:")
    print(f"   Precision: {np.mean(bootstrap_kendall_prec):.4f} [{kendall_prec_ci[0]:.4f}, {kendall_prec_ci[1]:.4f}]")
    print(f"   Recall:    {np.mean(bootstrap_kendall_rec):.4f} [{kendall_rec_ci[0]:.4f}, {kendall_rec_ci[1]:.4f}]")
    print(f"   F1:        {np.mean(bootstrap_kendall_f1):.4f} [{kendall_f1_ci[0]:.4f}, {kendall_f1_ci[1]:.4f}]")
    print("📉 RMSE:")
    print(f"   Precision: {np.mean(bootstrap_rmse_prec):.4f} [{rmse_prec_ci[0]:.4f}, {rmse_prec_ci[1]:.4f}]")
    print(f"   Recall:    {np.mean(bootstrap_rmse_rec):.4f} [{rmse_rec_ci[0]:.4f}, {rmse_rec_ci[1]:.4f}]")
    print(f"   F1:        {np.mean(bootstrap_rmse_f1):.4f} [{rmse_f1_ci[0]:.4f}, {rmse_f1_ci[1]:.4f}]")

    return {
        'model': model_name,
        'pearson_prec': pearson_prec[0],
        'pearson_rec': pearson_rec[0],
        'pearson_f1': pearson_f1[0],
        'spearman_prec': spearman_prec[0],
        'spearman_rec': spearman_rec[0],
        'spearman_f1': spearman_f1[0],
        'kendall_prec': kendall_prec[0],
        'kendall_rec': kendall_rec[0],
        'kendall_f1': kendall_f1[0],
        'rmse_prec': rmse_prec,
        'rmse_rec': rmse_rec,
        'rmse_f1': rmse_f1
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--factspotter-electra-path', type=Path, default=DEFAULT_FACTSPOTTER_ELECTRA_PATH,
                        help='Path to the original FactSpotter-ELECTRA weights (default: bundled checkpoints/fact_spotter_electra.pt).')
    parser.add_argument('--factspotter-deberta-small', action='store_true',
                        help='Also run the optional FactSpotter-DeBERTaV3-Small variant (not used for the paper FS row).')
    parser.add_argument('--factspotter-deberta-base', action='store_true',
                        help='Also run the optional FactSpotter-DeBERTaV3-Base variant (not used for the paper FS row).')
    parser.add_argument('--factspotter-deberta-large', action='store_true',
                        help='Also run the optional FactSpotter-DeBERTaV3-Large variant (not used for the paper FS row).')
    args = parser.parse_args()
    input_json = DEFAULT_INPUT_JSON
    template_file = DEFAULT_TEMPLATE_FILE
    factspotter_electra_path = args.factspotter_electra_path.expanduser().resolve()
    factspotter_electra_base = "google/electra-small-discriminator"
    use_factspotter_deberta_small = args.factspotter_deberta_small
    use_factspotter_deberta_base = args.factspotter_deberta_base
    use_factspotter_deberta_large = args.factspotter_deberta_large
    output_dir = ensure_dir(DEFAULT_OUTPUT_DIR)

    if not input_json.exists():
        raise FileNotFoundError(f"Missing input file: {input_json}")
    if not template_file.exists():
        raise FileNotFoundError(f"Missing template file: {template_file}")
    if not factspotter_electra_path.exists():
        raise FileNotFoundError(f"Missing FactSpotter-ELECTRA checkpoint: {factspotter_electra_path}")

    print(f"\n🎲 Random seed: {RANDOM_SEED}")

    # ========================================================================
    # RUN INFERENCE
    # ========================================================================
    results_monolr, results_nli, results_fs_electra, results_fs_deberta_small, results_fs_deberta_base, results_fs_deberta_large = run_inference(
        str(input_json), str(template_file),
        str(factspotter_electra_path), factspotter_electra_base,
        use_factspotter_deberta_small,
        use_factspotter_deberta_base,
        use_factspotter_deberta_large
    )

    # ========================================================================
    # COMPUTE CORRELATIONS
    # ========================================================================
    corr_monolr = compute_correlations_full(results_monolr, "MonoLR (William)")
    corr_nli = compute_correlations_full(results_nli, "NLI-Based (Dusek)")

    corr_fs_electra = None
    if results_fs_electra:
        corr_fs_electra = compute_correlations_recall_only(results_fs_electra, "FactSpotter-ELECTRA (RECALL ONLY)")

    corr_fs_deberta_small = None
    if results_fs_deberta_small:
        corr_fs_deberta_small = compute_correlations_recall_only(results_fs_deberta_small,
                                                                 "FactSpotter-DeBERTa-Small (RECALL ONLY)")

    corr_fs_deberta_base = None
    if results_fs_deberta_base:
        corr_fs_deberta_base = compute_correlations_recall_only(results_fs_deberta_base,
                                                                "FactSpotter-DeBERTa-Base (RECALL ONLY)")

    corr_fs_deberta_large = None
    if results_fs_deberta_large:
        corr_fs_deberta_large = compute_correlations_recall_only(results_fs_deberta_large,
                                                                 "FactSpotter-DeBERTa-Large (RECALL ONLY)")

    save_json(output_dir / "monolr_results.json", results_monolr)
    save_json(output_dir / "nli_results.json", results_nli)
    save_json(output_dir / "factspotter_electra_results.json", results_fs_electra)
    if use_factspotter_deberta_small:
        save_json(output_dir / "factspotter_deberta_small_results.json", results_fs_deberta_small)
    if use_factspotter_deberta_base:
        save_json(output_dir / "factspotter_deberta_base_results.json", results_fs_deberta_base)
    if use_factspotter_deberta_large:
        save_json(output_dir / "factspotter_deberta_large_results.json", results_fs_deberta_large)

    correlations = {
        "MonoLR": corr_monolr,
        "NLI": corr_nli,
        "FactSpotter-ELECTRA": corr_fs_electra,
        "FactSpotter-DeBERTa-Small": corr_fs_deberta_small,
        "FactSpotter-DeBERTa-Base": corr_fs_deberta_base,
        "FactSpotter-DeBERTa-Large": corr_fs_deberta_large,
    }
    save_json(output_dir / "plm_baselines_summary.json", {
        "input_json": input_json,
        "template_file": template_file,
        "factspotter_electra_path": factspotter_electra_path,
        "correlations": correlations,
    })
    log_path = write_summary_log(output_dir, {
        "input_json": input_json,
        "template_file": template_file,
        "factspotter_electra_path": factspotter_electra_path,
    }, correlations)

    print("\n" + "=" * 70)
    print(f"Saved baseline outputs to: {output_dir}")
    print(f"Saved baseline log to: {log_path}")
    print("✅ Done!")
    print("=" * 70)
