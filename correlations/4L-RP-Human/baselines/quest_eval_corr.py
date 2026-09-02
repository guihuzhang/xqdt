#!/usr/bin/env python3
"""
QuestEval Baseline - Extracting Separate P/R/F1
Based on your previous WebNLG script
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
from tqdm import tqdm

from questeval_metric import QuestEval

# Add questeval to path if needed
# sys.path.append('path/to/questeval')

# Random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

SCRIPT_DIR = Path(__file__).resolve().parent
CORRELATION_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT_JSON = CORRELATION_ROOT / "by_language" / "english.json"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results"


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


def write_summary_log(output_dir: Path, input_json: Path, correlation: dict):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"questeval_evaluation_{timestamp}.log"
    lines = [
        "4L-RP-Human QuestEval evaluation",
        f"input_json: {input_json}",
        f"output_dir: {output_dir}",
        "",
        "QuestEval",
        f"  Pearson:  P={correlation['pearson_prec']:.4f} R={correlation['pearson_rec']:.4f} F1={correlation['pearson_f1']:.4f}",
        f"  Spearman: P={correlation['spearman_prec']:.4f} R={correlation['spearman_rec']:.4f} F1={correlation['spearman_f1']:.4f}",
        f"  Kendall:  P={correlation['kendall_prec']:.4f} R={correlation['kendall_rec']:.4f} F1={correlation['kendall_f1']:.4f}",
        "",
    ]
    log_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return log_path


# ============================================================================
# QUESTEVAL P/R/F1 EXTRACTION
# ============================================================================

def extract_prf_from_questeval(questeval_model, hypothesis: str, source: list) -> tuple:
    """
    Extract separate P/R/F1 from QuestEval logs

    Args:
        questeval_model: Loaded QuestEval instance
        hypothesis: Generated text
        source: List of triples

    Returns:
        precision, recall, f1
    """
    try:
        # Get preprocessed source text
        if questeval_model.src_preproc_pipe:
            src_text = questeval_model.src_preproc_pipe(source)
        else:
            src_text = ' '.join(source)

        # Load logs
        hyp_log = questeval_model.open_log_from_text(hypothesis)
        src_log = questeval_model.open_log_from_text(src_text)

        # Compute precision: hypothesis -> source
        # (How much of hypothesis can be answered by source)
        precision = questeval_model._base_score(src_log, hyp_log)

        # Compute recall: source -> hypothesis
        # (How much of source is covered by hypothesis)
        recall = questeval_model._base_score(hyp_log, src_log)

        # F1
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        return precision, recall, f1

    except Exception as e:
        print(f"⚠️  Error in extract_prf: {e}")
        return 0.0, 0.0, 0.0


def compute_questeval_prf_batch(questeval_model, hypotheses: list, sources: list, batch_size: int = 16):
    """
    Compute QuestEval P/R/F1 for a batch of examples

    Args:
        questeval_model: Loaded QuestEval instance
        hypotheses: List of generated texts
        sources: List of source triples (each is a list of triples)
        batch_size: Batch size for QuestEval

    Returns:
        precisions, recalls, f1s (three lists)
    """
    print("\n" + "─" * 70)
    print("Step 1: Running QuestEval to populate logs...")
    print("─" * 70)

    # Step 1: Run QuestEval to populate logs
    # This is necessary to generate all the QA pairs and cache them
    d_score = questeval_model.corpus_questeval(
        hypothesis=hypotheses,
        sources=sources,
        list_references=None,
        batch_size=batch_size
    )

    print(f"✅ QuestEval completed! Corpus score: {d_score['corpus_score']:.4f}")

    print("\n" + "─" * 70)
    print("Step 2: Extracting separate P/R/F1 from logs...")
    print("─" * 70)

    # Step 2: Extract P/R/F1 from logs
    precisions = []
    recalls = []
    f1s = []

    for hyp, src in tqdm(zip(hypotheses, sources), total=len(hypotheses), desc="Extracting P/R/F1"):
        prec, rec, f1 = extract_prf_from_questeval(questeval_model, hyp, src)
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    print(f"\n✅ Extraction completed!")
    print(f"   Avg Precision: {np.mean(precisions):.4f}")
    print(f"   Avg Recall:    {np.mean(recalls):.4f}")
    print(f"   Avg F1:        {np.mean(f1s):.4f}")

    return precisions, recalls, f1s


# ============================================================================
# DATA PROCESSING
# ============================================================================

def format_source_for_questeval(graph_str: str) -> list:
    """
    Convert graph string to list of triples for QuestEval

    Input: "subject | predicate | object<br>subject2 | predicate2 | object2"
    Output: ["subject | predicate | object", "subject2 | predicate2 | object2"]
    """
    triples = [t.strip() for t in graph_str.split('<br>') if t.strip()]
    return triples


# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================

def compute_correlations_questeval(results: list, model_name: str,
                                   n_bootstrap: int = 1000, confidence: float = 0.95):
    """Compute correlations with bootstrap for QuestEval"""
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

    # Bootstrap
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
        'pearson_prec': np.mean(bootstrap_pearson_prec),
        'pearson_rec': np.mean(bootstrap_pearson_rec),
        'pearson_f1': np.mean(bootstrap_pearson_f1),
        'spearman_prec': np.mean(bootstrap_spearman_prec),
        'spearman_rec': np.mean(bootstrap_spearman_rec),
        'spearman_f1': np.mean(bootstrap_spearman_f1),
        'kendall_prec': np.mean(bootstrap_kendall_prec),
        'kendall_rec': np.mean(bootstrap_kendall_rec),
        'kendall_f1': np.mean(bootstrap_kendall_f1),
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    input_json = DEFAULT_INPUT_JSON
    use_cuda = True
    use_cache = True
    batch_size = 16
    output_dir = ensure_dir(DEFAULT_OUTPUT_DIR)

    if not input_json.exists():
        raise FileNotFoundError(f"Missing input file: {input_json}")

    print(f"\n🎲 Random seed: {RANDOM_SEED}")

    # ========================================================================
    # LOAD DATA
    # ========================================================================
    print(f"\n📂 Loading: {input_json}")
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"📊 Total samples: {len(data)}")

    # ========================================================================
    # LOAD QUESTEVAL
    # ========================================================================
    print("\n" + "─" * 70)
    print("Loading QuestEval model...")
    print("─" * 70)
    print("⚠️  This may take a while on first run (downloading models)...")

    questeval = QuestEval(
        task="data2text",
        language="en",
        use_cache=use_cache,
        no_cuda=not use_cuda
    )

    print("✅ QuestEval loaded!")

    # ========================================================================
    # PREPARE DATA
    # ========================================================================
    print("\n" + "─" * 70)
    print("Preparing data...")
    print("─" * 70)

    hypotheses = [item['text'] for item in data]
    sources = [format_source_for_questeval(item['graph']) for item in data]

    print(f"✅ Prepared {len(hypotheses)} examples")

    # ========================================================================
    # COMPUTE QUESTEVAL P/R/F1
    # ========================================================================
    precisions, recalls, f1s = compute_questeval_prf_batch(
        questeval,
        hypotheses,
        sources,
        batch_size=batch_size
    )

    # ========================================================================
    # ORGANIZE RESULTS
    # ========================================================================
    print("\n" + "─" * 70)
    print("Organizing results...")
    print("─" * 70)

    results_questeval = []
    for item, prec, rec, f1 in zip(data, precisions, recalls, f1s):
        results_questeval.append({
            'id': item['id'],
            'question_id': item['question_id'],
            'text': item['text'],
            'graph': item['graph'],
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'human_precision_list': item.get('precision_list', []),
            'human_recall_list': item.get('recall_list', []),
        })

    # ========================================================================
    # COMPUTE CORRELATIONS
    # ========================================================================
    corr_questeval = compute_correlations_questeval(results_questeval, "QuestEval")
    save_json(output_dir / "questeval_results.json", results_questeval)
    save_json(output_dir / "questeval_correlation.json", corr_questeval)
    save_json(output_dir / "questeval_summary.json", {
        "input_json": input_json,
        "correlation": corr_questeval,
    })
    log_path = write_summary_log(output_dir, input_json, corr_questeval)

    print("\n" + "=" * 70)
    print(f"Saved QuestEval outputs to: {output_dir}")
    print(f"Saved QuestEval log to: {log_path}")
    print("✅ Done!")
    print("=" * 70)
