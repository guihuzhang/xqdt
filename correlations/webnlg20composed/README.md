# WebNLG 2020 composed correlations

This directory contains WebNLG 2020 inputs, human ratings, saved predictions,
and scoring code. It evaluates how well metric precision, recall, and F1
agree with human judgments across 177 inputs and 16 generation systems.

The results below have been updated to correct a bug identified after paper
submission.

## Results

Values are bootstrap-mean Spearman correlations × 100, rounded to one decimal.
Columns compare metric precision, recall, and F1 with the
[composed human targets](#data-and-human-targets), at system and text level.

| Model | `P_m` vs `P_a` `rho_sys` | `P_m` vs `P_a` `rho_txt` | `R_m` vs `R_a` `rho_sys` | `R_m` vs `R_a` `rho_txt` | `F1_m` vs `F1_a` `rho_sys` | `F1_m` vs `F1_a` `rho_txt` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DQE | 66.3 | 56.2 | 72.3 | 56.6 | 74.3 | 62.8 |
| MonoLR | 77.1 | 58.5 | 90.8 | 61.0 | 85.7 | 65.1 |
| NLI | 66.0 | 62.4 | 85.3 | 62.1 | 80.5 | 63.2 |
| FS | --- | --- | 89.5 | 63.7 | --- | --- |
| G3-1B | 79.7 | 62.3 | 92.8 | 64.0 | 89.3 | 63.8 |
| G3-4B | 69.9 | 62.9 | 88.6 | 64.4 | 81.6 | 65.5 |
| G3-12B | 69.1 | 62.5 | 88.6 | 64.4 | 81.8 | 64.5 |
| Q3-0.6B | 77.9 | 63.3 | 92.7 | 63.4 | 87.7 | 64.2 |
| Q3-1.7B | 78.7 | 62.1 | 91.7 | 64.2 | 88.1 | 64.6 |
| Q3-4B | 65.7 | 61.9 | 83.4 | 64.3 | 76.0 | 64.0 |
| Q3-8B | 75.2 | 62.8 | 90.2 | 64.5 | 85.2 | 63.9 |
| Q3-14B | 73.2 | 60.5 | 91.6 | 63.3 | 86.0 | 64.2 |
| L3.2-1B | 82.1 | 63.1 | 93.6 | 64.2 | 90.5 | 63.7 |
| L3.2-3B | 67.9 | 62.5 | 86.0 | 63.1 | 79.2 | 63.9 |
| L3.1-8B | 72.2 | 62.8 | 88.0 | 63.2 | 81.6 | 63.4 |

G3, Q3, and L3 denote Gemma 3, Qwen 3, and Llama 3. Gemma3-270M results
are bundled but not reported in Table 6. DQE uses `metric_results/questeval_p`,
`questeval_r`, and `questeval` for P, R, and F1. FS uses
`metric_results/fact-spotter` for R only, hence its `---` cells.

### Prompted baselines

| Model | `P_m` vs `P_a` `rho_sys` | `P_m` vs `P_a` `rho_txt` | `R_m` vs `R_a` `rho_sys` | `R_m` vs `R_a` `rho_txt` | `F1_m` vs `F1_a` `rho_sys` | `F1_m` vs `F1_a` `rho_txt` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gemma3-27B (prompted) | 60.8 | 48.0 | 56.3 | 52.1 | 58.2 | 53.2 |
| Qwen3-32B (prompted) | 59.0 | 56.6 | 66.0 | 58.9 | 62.6 | 53.6 |
| Llama3.3-70B (prompted) | 65.8 | 57.2 | 73.5 | 54.5 | 70.4 | 54.4 |
| GPT-5.1 (prompted) | 71.2 | 61.9 | 80.4 | 62.3 | 76.9 | 62.4 |

Both tables come from `correlation_logs/webnlg2020/composed_corr_results_*.json`:
`baselines`, `gemma`, `qwen`, and `llama` for the first table, and
`prompted` for all four prompted models.

## Environment and scoring

Use Python 3.11 and run the commands from the repository root. Install the
scoring dependencies, then recompute correlations from the saved example-level
scores:

```bash
python -m pip install -r correlations/webnlg20composed/requirements-correlation.txt

bash correlations/webnlg20composed/scripts/run_composed_corr_baselines.sh
bash correlations/webnlg20composed/scripts/run_composed_corr_gemma.sh
bash correlations/webnlg20composed/scripts/run_composed_corr_qwen.sh
bash correlations/webnlg20composed/scripts/run_composed_corr_llama.sh

python correlations/webnlg20composed/corr_webnlg20_composed.py \
  --metric-results-dir correlations/webnlg20composed/prompted_llm_metric_results \
  --output-suffix prompted
```

These commands read the bundled metric scores and write the five correlation
JSON files listed above, with matching logs, under `correlation_logs/webnlg2020/`.
The same commands produce the full Pearson, Spearman, and Kendall results
for precision, recall, and F1, including the appendix results and
Gemma3-270M. The JSON files also contain bootstrap bounds and RMSE.

## Rebuild metric files from parsed predictions

This step converts the saved `parsed_errors` into per-example precision,
recall, and F1 scores in `dt_*_{p,r,f1}/` folders. For the XQDT adapters:

```bash
python correlations/webnlg20composed/build_metric_results_from_xqdt.py \
  --triples-file correlations/webnlg20composed/webnlg2020_inputs/triples_modified.txt
```

For the four prompted baselines:

```bash
python correlations/webnlg20composed/build_metric_results_from_xqdt.py \
  --results-dir correlations/webnlg20composed/prompted_llm_results \
  --output-dir correlations/webnlg20composed/prompted_llm_metric_results
```

Then rerun the corresponding [correlation commands](#environment-and-scoring).

## Fresh inference with published adapters

For model inference, use Python 3.11 and install the verifier dependencies:

```bash
python -m pip install -r verifier_train_eval_webnlg/requirements-py311.txt
```

The four entry points group adapters by model size and default to the bundled
`webnlg2020_inputs/team-generations/` and `webnlg2020_inputs/triples_modified.txt`.
Use `--team-dir` and `--triples-file` to select other inputs.

```bash
for size in 1 4 8 12; do
  USE_HF=1 python "correlations/webnlg20composed/code/obtain_output${size}.py" \
    --output-dir correlations/webnlg20composed/xqdt_results_rerun \
    --skip-existing
done
```

Score the new predictions and compute their correlations:

```bash
python correlations/webnlg20composed/build_metric_results_from_xqdt.py \
  --results-dir correlations/webnlg20composed/xqdt_results_rerun \
  --triples-file correlations/webnlg20composed/webnlg2020_inputs/triples_modified.txt \
  --output-dir correlations/webnlg20composed/metric_results_rerun
python correlations/webnlg20composed/corr_webnlg20_composed.py \
  --eval-results-dir correlations/webnlg20composed/xqdt_results_rerun \
  --metric-results-dir correlations/webnlg20composed/metric_results_rerun \
  --log-dir correlations/webnlg20composed/correlation_logs/rerun
```

### Published adapters

- `Gemma3-270M` -> [`Loria-MosAIk/xqdt-webnlg-gemma3-270m`](https://huggingface.co/Loria-MosAIk/xqdt-webnlg-gemma3-270m)
- `Gemma3-1B` -> [`Loria-MosAIk/xqdt-webnlg-gemma3-1b`](https://huggingface.co/Loria-MosAIk/xqdt-webnlg-gemma3-1b)
- `Gemma3-4B` -> [`Loria-MosAIk/xqdt-webnlg-gemma3-4b`](https://huggingface.co/Loria-MosAIk/xqdt-webnlg-gemma3-4b)
- `Gemma3-12B` -> [`Loria-MosAIk/xqdt-webnlg-gemma3-12b`](https://huggingface.co/Loria-MosAIk/xqdt-webnlg-gemma3-12b)
- `Qwen3-0.6B` -> [`Loria-MosAIk/xqdt-webnlg-qwen3-0.6b`](https://huggingface.co/Loria-MosAIk/xqdt-webnlg-qwen3-0.6b)
- `Qwen3-1.7B` -> [`Loria-MosAIk/xqdt-webnlg-qwen3-1.7b`](https://huggingface.co/Loria-MosAIk/xqdt-webnlg-qwen3-1.7b)
- `Qwen3-4B` -> [`Loria-MosAIk/xqdt-webnlg-qwen3-4b`](https://huggingface.co/Loria-MosAIk/xqdt-webnlg-qwen3-4b)
- `Qwen3-8B` -> [`Loria-MosAIk/xqdt-webnlg-qwen3-8b`](https://huggingface.co/Loria-MosAIk/xqdt-webnlg-qwen3-8b)
- `Qwen3-14B` -> [`Loria-MosAIk/xqdt-webnlg-qwen3-14b`](https://huggingface.co/Loria-MosAIk/xqdt-webnlg-qwen3-14b)
- `Llama3.2-1B` -> [`Loria-MosAIk/xqdt-webnlg-llama3.2-1b`](https://huggingface.co/Loria-MosAIk/xqdt-webnlg-llama3.2-1b)
- `Llama3.2-3B` -> [`Loria-MosAIk/xqdt-webnlg-llama3.2-3b`](https://huggingface.co/Loria-MosAIk/xqdt-webnlg-llama3.2-3b)
- `Llama3.1-8B` -> [`Loria-MosAIk/xqdt-webnlg-llama3.1-8b`](https://huggingface.co/Loria-MosAIk/xqdt-webnlg-llama3.1-8b)

### Optional: rerun traditional baselines

```bash
python correlations/webnlg20composed/baselines/run_legacy_baselines.py \
  --models monolr nli factspotter dqe \
  --output-dir correlations/webnlg20composed/baseline_rerun
```

This requires baseline dependencies and weights. It writes regenerated scores
to `baseline_rerun/` and compares them with the bundled scores.
Use `--help` for model and input options.

## Data and human targets

- `human-annotations/human_eval_webnlg_20.json`: input graphs, references,
  system outputs, and human correctness (CR), relevance (RV), and
  data-coverage (CV) ratings.
- `xqdt_results/` and `prompted_llm_results/`: saved verifier responses and
  parsed errors.
- `metric_results/` and `prompted_llm_metric_results/`: example-level scores.
- `webnlg2020_inputs/triples_modified.txt` and `team-generations/`: input
  graphs and the outputs of 16 systems. The files have 178 rows to preserve
  entry IDs; entry 106 has no human ratings and is excluded, leaving
  177 evaluated graphs and 2,832 texts.

Human ratings are scaled to [0, 1] before composition:

- `P_a = CR × RV`
- `R_a = CV`
- `F1_a = harmonic_mean(P_a, R_a)`

The evaluation uses 1,000 bootstrap replicates over input graphs with seed 10.
System-level correlations compare system means, retaining replicates with
`p < 0.05`. Text-level correlations average per-input correlations across
systems, retaining those with `p < 0.05` within each replicate.
The JSON results include retained replicate counts and the 5th and 95th
percentiles, which form a 90% bootstrap interval. They also report pooled
correlations over the 2,832 outputs separately.

## QuestEval source

`baselines/questeval_metric.py` and `baselines/quest_utils.py` adapt
[QuestEval](https://github.com/ThomasScialom/QuestEval) for this
workflow, deriving directional P/R scores from source/hypothesis
question-answering logs.
The [MIT license](../../third_party_licenses/QuestEval-LICENSE.txt) is included.
