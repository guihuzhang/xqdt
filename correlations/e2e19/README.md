# E2E19 Human-Rating Evaluation

This directory contains the inputs, cached predictions, and scoring scripts for
the E2E human-rating experiments in XQDT.

## Included artifacts

- `human_ratings/converted.json`: E2E inputs, system outputs, and human ratings
- `human_ratings/excluded_workers.txt`: annotator IDs excluded when computing the paper results
- `human_ratings/annotator_quality/`: annotator-quality flags and associated cases
- `xqdt_results_repaired/`: cached XQDT predictions and a generated Table 7 summary
- `prompted_llm_results/`: final prompted-LLM outputs
- `baselines/baseline_scores_e2e19.json`: cached NLI, DQE, and FactSpotter scores
- `paired_cluster_bootstrap_results/`: saved paired-bootstrap results

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

The bundled predictions reproduce all twelve XQDT rows reported in Table 7.

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

For Qwen inference, run with `USE_HF=1`.

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
