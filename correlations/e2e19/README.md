# E2E19 Human-Rating Evaluation

This directory contains the inputs, cached predictions, and scoring scripts for
the E2E human-rating experiments in XQDT.

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

Annotator identifiers are stable within this release. The annotation-quality
categories and filtering decisions are documented in
`human_ratings/annotator_quality/README.md`.

## Reproduce the saved XQDT rows

```bash
python correlations/e2e19/evaluate_coarse_labels.py \
  --results-dir correlations/e2e19/xqdt_results_repaired \
  --output-dir correlations/e2e19/xqdt_results_repaired
```

The scorer writes:

- `table7_majority_lenient_summary.json`
- `coarse_label_eval_majority_lenient_YYYYMMDD_HHMMSS.log`

Paper-ready values are stored under
`models.<model_name>.paper_row_percent` in the summary JSON.

The bundled predictions cover twelve XQDT models. The table below reports scores recomputed from these outputs.

### Expected XQDT F1 scores

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

## Reproduce the saved baseline rows

```bash
python correlations/e2e19/baselines/evaluate_coarse_labels_baselines.py \
  --scores-file correlations/e2e19/baselines/baseline_scores_e2e19.json \
  --output-dir correlations/e2e19/baselines
```

This reproduces the NLI, DQE, and FactSpotter rows reported in Table 7.

## Reproduce the correlation tables

Recompute all twelve XQDT models (P/R/F1), the traditional baselines, and the
six reference-based metrics from their example-level caches:

```bash
python correlations/e2e19/score_cached_correlations.py
```

This does not run model inference or change the input caches. It writes
`correlation_results/correlations.json` with full-precision coefficients and
`correlation_results/correlations.md` with the readable table. Each row includes
the number of valid evaluation pairs. Use `--output-dir` to choose another
output directory.

The inputs contain 6,012 MR--system pairs. After human-rating filtering, 5,515
pairs remain. The historical DQE cache has scores for 4,280 input pairs, of
which 3,896 have retained human ratings; its correlations use those 3,896 pairs.
The original experiment log reports the same count. Missing DQE scores are
excluded, not treated as zero.

The generated table rounds full-precision coefficients directly. Across the
52 rows, 301 of the 312 coefficients match the published one-decimal values;
the remaining 11 differ by 0.1 and are consistent with rounding the historical
four-decimal coefficients before converting to percentages. The underlying
cached predictions are unchanged.

### Reference-based metrics only

The six reference-based metrics in the appendix can be rescored from the
bundled cache without running model inference:

```bash
python correlations/e2e19/baselines/ref_baseline_eval_e2e19.py
```

The script reuses the example-level scores, applies the human-rating filters,
and writes the text- and system-level correlations to
`baselines/results/ref_baseline_eval_e2e19.log`. The cache and reference file
were exported unchanged from the original experiment repository at commit
`18ee56f`.

## Score a prompted-LLM output

```bash
python correlations/e2e19/score_prompted_llm_e2e.py \
  --results correlations/e2e19/prompted_llm_results/Gemma3-27B_results.json
```

The prompted-LLM directory contains the final JSON outputs used for scoring.
Incremental JSONL generation checkpoints are not included.

## Re-run inference

The `llm_eval_*.py` scripts write final P/R/F1 values directly, with:

- `FP = extra + incorrect`
- `FN = missing + incorrect`

The E2E LoRA adapters are available in the
[E2E model collection](https://huggingface.co/collections/Loria-MosAIk/xqdt-e2e-models-6a94697d5be8186d7795da71).
The inference scripts download the corresponding published adapters from
Hugging Face.

Inference respects the externally configured `CUDA_VISIBLE_DEVICES`. Failed
batches, missing responses and unrecognized output formats raise an error;
incomplete predictions are not written as completed results.

## Environment

Full inference shares the WebNLG verifier dependencies in
`verifier_train_eval_webnlg/requirements-py311.txt` and needs a compatible
PyTorch/CUDA environment. Cached correlation reproduction needs NumPy and
SciPy; Table 7 scoring requires NumPy and the bundled
worker-filtering helper.

## Repaired score fields

`xqdt_results_repaired/` contains saved model responses and parsed errors with
recomputed P/R/F1. An incorrect unit contributes to both FP and FN:
`FP = extra + incorrect`, `FN = missing + incorrect`.
Earlier scores are retained in `*_old` fields where present. Across the twelve
caches, 498 sample/model records have different old and current P/R/F1 values.

## WebNLG-only transfer

The twelve WebNLG-only prediction caches are bundled under
`verifier_train_eval_e2e19/cross_dataset_results/webnlg_to_e2e_human/`.
Each contains the same 6,012 MR/system pairs as the E2E-trained caches.
`table19_e2e_in_vs_cross_domain.py` uses these inputs for the training-setting
comparison; `paired_cluster_bootstrap_transfer.py` uses the 4B and 8B pairs.
Their command-line options, including output locations, are available with
`--help`. Saved bootstrap results are in `paired_cluster_bootstrap_results/`.

## Annotator-quality analysis

The publication-time filtering strategy is:

- exclude annotators in groups A, C, D, E, and F
- retain group B annotators while removing individual scores above 100

The six diagnostic groups, annotator-level flags, and flagged rating cases are
provided under `human_ratings/annotator_quality/`.

## Sanity checks

```bash
ls correlations/e2e19/human_ratings
ls correlations/e2e19/xqdt_results_repaired
ls correlations/e2e19/prompted_llm_results
ls correlations/e2e19/baselines
```
