# E2E19 Human-Rating Evaluation

This directory contains the inputs, cached predictions, and scoring scripts for
the E2E human-rating experiments in XQDT.

## Expected XQDT F1 scores

The following values provide a quick check of the reproduced results. Full
precision, recall, and F1 scores are available in
`table7_majority_lenient_summary.json`.

| Model | Is-ok F1 | Has-missing F1 | Has-added F1 |
| --- | ---: | ---: | ---: |
| Gemma3-270M | 88.6 | 74.6 | 24.3 |
| Gemma3-1B | 89.1 | 75.4 | 23.8 |
| Gemma3-4B | 89.2 | 75.6 | 20.1 |
| Gemma3-12B | 89.4 | 76.0 | 21.9 |
| Qwen3-0.6B | 89.3 | 75.7 | 20.8 |
| Qwen3-1.7B | 89.3 | 75.7 | 23.4 |
| Qwen3-4B | 89.4 | 75.8 | 21.1 |
| Qwen3-8B | 89.4 | 75.7 | 23.3 |
| Qwen3-14B | 89.1 | 75.5 | 18.7 |
| Llama3.2-1B | 89.1 | 75.3 | 24.1 |
| Llama3.2-3B | 89.4 | 76.0 | 21.9 |
| Llama3.1-8B | 89.3 | 76.0 | 22.3 |

## Scoring setup

Use Python 3.11 and install the lightweight scoring dependencies:

```bash
python -m pip install -r correlations/requirements.txt
```

Run all commands in this README from the repository root.

## Included artifacts

- `human_ratings/converted.json`: E2E inputs, system outputs, and human ratings
- `human_ratings/excluded_workers.txt`: annotator IDs excluded when computing the paper results
- `human_ratings/annotator_quality/`: annotator-quality flags and associated cases
- `xqdt_results_repaired/`: cached XQDT predictions and a generated Table 7 summary
- `prompted_llm_results/`: final prompted-LLM outputs
- `baselines/baseline_scores_e2e19.json`: cached NLI, MonoLR, DQE, and FactSpotter scores
- `baselines/ref_baseline_scores_e2e19.json`: example-level BLEU, METEOR, PARENT, BERTScore, BARTScore, and BLEURT scores for 6,012 system outputs
- `human_ratings/refs_e2e19.json`: reference texts grouped by 630 meaning representations
- `paired_cluster_bootstrap_results/`: saved paired-bootstrap results
- `paired_cluster_bootstrap_transfer.py`: paired cluster-bootstrap analysis for the WebNLG-only versus WebNLG+E2E training comparison

Annotation issues, cleaning rules, and flagged cases are documented in
[Annotation quality](human_ratings/annotator_quality/README.md).

## Reproduce the saved XQDT rows

```bash
python correlations/e2e19/evaluate_coarse_labels.py \
  --results-dir correlations/e2e19/xqdt_results_repaired \
  --output-dir /tmp/xqdt-e2e19/coarse-labels
```

The scorer writes:

- `table7_majority_lenient_summary.json`
- `coarse_label_eval_majority_lenient_YYYYMMDD_HHMMSS.log`

Paper-ready values are stored under
`models.<model_name>.paper_row_percent` in the summary JSON.

The bundled predictions cover twelve XQDT models. The table below reports scores recomputed from these outputs.

## Reproduce the saved baseline rows

```bash
python correlations/e2e19/baselines/evaluate_coarse_labels_baselines.py \
  --scores-file correlations/e2e19/baselines/baseline_scores_e2e19.json \
  --output-dir /tmp/xqdt-e2e19/baselines
```

This reproduces the NLI, DQE, and FactSpotter rows reported in Table 7.

## Reproduce the correlation tables

Recompute all twelve XQDT models (P/R/F1), the traditional baselines, and the
six reference-based metrics from their example-level caches:

```bash
python correlations/e2e19/score_cached_correlations.py \
  --output-dir /tmp/xqdt-e2e19/correlations
```

The command writes
`correlations.json` with full-precision coefficients and `correlations.md` with
the readable table in the chosen output directory. Each row includes the number
of valid evaluation pairs.

The inputs contain 6,012 MR--system pairs. After human-rating filtering, 5,515
pairs remain. The DQE cache has scores for 4,280 input pairs, of
which 3,896 have retained human ratings; its correlations use those 3,896 pairs.
Pairs without DQE scores are excluded.

Coefficients are computed at full precision and rounded to one decimal.

### Reference-based metrics only

The six reference-based metrics in the appendix can be rescored from the
bundled cache without running model inference:

```bash
python correlations/e2e19/baselines/ref_baseline_eval_e2e19.py
```

The script reuses the example-level scores, applies the human-rating filters,
and writes the text- and system-level correlations to
`baselines/results/ref_baseline_eval_e2e19.log`.

## Score a prompted-LLM output

The cached outputs cover Gemma 3 27B, Qwen3 32B, and Llama 3.3 70B.

```bash
mkdir -p /tmp/xqdt-e2e19/prompted
python correlations/e2e19/score_prompted_llm_e2e.py \
  --results correlations/e2e19/prompted_llm_results/Gemma3-27B_results.json \
  --output /tmp/xqdt-e2e19/prompted/Gemma3-27B_correlation.json
```

## Re-run inference

The `llm_eval_*.py` scripts write final P/R/F1 values directly, with:

- `FP = extra + incorrect`
- `FN = missing + incorrect`

The E2E LoRA adapters are available in the
[E2E model collection](https://huggingface.co/collections/Loria-MosAIk/xqdt-e2e-models-6a94697d5be8186d7795da71).
The inference scripts download the corresponding published adapters from
Hugging Face.

Inference uses the GPUs selected by `CUDA_VISIBLE_DEVICES`.

## Environment

Full inference shares the WebNLG verifier dependencies in
[`verifier_train_eval_webnlg/requirements-py311.txt`](../../verifier_train_eval_webnlg/requirements-py311.txt)
and needs a compatible PyTorch/CUDA environment:

```bash
python -m pip install -r verifier_train_eval_webnlg/requirements-py311.txt
```

## Repaired score fields

`xqdt_results_repaired/` contains saved model responses and parsed errors with
recomputed P/R/F1. An incorrect unit contributes to both FP and FN:
`FP = extra + incorrect`, `FN = missing + incorrect`.
Earlier scores are retained in `*_old` fields where present. Across the twelve
caches, 498 sample/model records have different old and current P/R/F1 values.

## WebNLG-only transfer

This experiment compares WebNLG-only and WebNLG+E2E training on the same
human-rated E2E outputs, using correlations and coarse-label F1.

The twelve WebNLG-only prediction caches are stored under
`verifier_train_eval_e2e19/cross_dataset_results/webnlg_to_e2e_human/`.
Each contains the same 6,012 MR/system pairs as the WebNLG+E2E caches.

Recompute F1 correlations for all twelve models under WebNLG+E2E and
WebNLG-only training:

```bash
python correlations/e2e19/table19_e2e_in_vs_cross_domain.py \
  --output-dir /tmp/xqdt-e2e19/transfer
```

This writes `table19_in_vs_cross_domain_summary.json` for all twelve models and
`table19_in_vs_cross_domain_qwen.md` for the five Qwen models, using the same
human-rating filters as the correlation tables above.

For paired 95% confidence intervals comparing WebNLG-only and WebNLG+E2E
training on Qwen3-4B and Qwen3-8B:

```bash
python correlations/e2e19/paired_cluster_bootstrap_transfer.py \
  --repo-root . \
  --models Qwen3-4B Qwen3-8B \
  --n-bootstrap 10000 \
  --seed 42 \
  --chunk-size 250 \
  --output-root /tmp/xqdt-e2e19/transfer-bootstrap
```

Coarse-label F1 and text-level correlations use paired MR-cluster resampling;
system-level correlations use paired system resampling. Differences are
WebNLG-only minus WebNLG+E2E. Each run creates a timestamped output
directory. Existing results are in `paired_cluster_bootstrap_results/`.

### Saved bootstrap results

The saved run uses 10,000 resamples, seed 42, and percentile 95% confidence
intervals. Selected results are shown below; scores and correlations are
multiplied by 100.

| Model | Measure | WebNLG+E2E | WebNLG-only | Difference | 95% CI |
| --- | --- | ---: | ---: | ---: | --- |
| Qwen3-4B | Is-ok F1 | 89.36 | 89.83 | +0.46 | [+0.20, +0.73] |
| Qwen3-4B | Has-missing F1 | 75.82 | 76.42 | +0.60 | [+0.08, +1.12] |
| Qwen3-4B | Has-added F1 | 21.09 | 23.94 | +2.86 | [-4.20, +9.55] |
| Qwen3-4B | Text-level Spearman | 51.52 | 51.39 | -0.13 | [-1.08, +0.76] |
| Qwen3-8B | Is-ok F1 | 89.39 | 89.22 | -0.17 | [-0.63, +0.21] |
| Qwen3-8B | Has-missing F1 | 75.74 | 75.22 | -0.52 | [-1.46, +0.28] |
| Qwen3-8B | Has-added F1 | 23.33 | 20.71 | -2.61 | [-9.63, +4.26] |
| Qwen3-8B | Text-level Spearman | 51.38 | 51.21 | -0.18 | [-0.98, +0.60] |

The complete results are available as
[JSON](paired_cluster_bootstrap_results/20260825T215729Z/results.json) and
[CSV](paired_cluster_bootstrap_results/20260825T215729Z/results.csv), with a
[LaTeX table](paired_cluster_bootstrap_results/20260825T215729Z/table.tex) for
coarse-label F1 and text-level correlations.

## Annotation issues and cleaning

We use the E2E NLG Challenge human evaluation described by
[Dušek, Novikova and Rieser (2020)](https://arxiv.org/abs/1901.07931), with
system-generated restaurant descriptions, their input meaning representations,
and human ratings released in the [original repository](https://github.com/tuetschek/e2e-eval).

The quality scores use a 0–100 scale. While preparing the XQDT evaluation,
we found problems in some of these ratings, including scores above 100,
repeated-digit entries, and disagreement between numeric scores and error
labels. Some annotators also used a very limited set of scores or always or
almost always gave 100.

The scoring scripts exclude the 23 annotators in
`human_ratings/excluded_workers.txt` and remove individual scores above 100
among the remaining annotators. For correlations, they average the remaining
quality ratings for each input–output pair. Pairs without any remaining
ratings are omitted, leaving 5,515 of the original 6,012 pairs. The source
ratings remain in `human_ratings/converted.json`. The
[cleaning notes](human_ratings/annotator_quality/README.md) describe the six
groups, examples, and files used to inspect these cases.
