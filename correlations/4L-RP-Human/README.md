# 4L-RP-Human Correlation Workflow

This directory contains the human annotations, example-level model predictions, and evaluation code used for the 4L-RP-Human experiments in XQDT.

4L-RP-Human provides human precision and recall judgements for data-text pairs, allowing us to directly measure how well automatic metrics agree with human assessments of data-text alignment. The paper reports results on the English subset, while this release also includes processed annotations for eight languages.

Cached outputs are provided for XQDT, prompted LLM baselines, and four traditional baselines. The XQDT adapters are available in the [WebNLG model collection](https://huggingface.co/collections/Loria-MosAIk/xqdt-webnlg-models-6a94697a0582edd20ad3df04).

## CPU scoring setup

Use Python 3.11 and install the lightweight scoring dependencies:

```bash
python -m pip install -r correlations/requirements.txt
```

Run all commands in this README from the repository root. Cached scoring needs
no GPU or model downloads; full inference uses the [verifier requirements](../../verifier_train_eval_webnlg/requirements-py311.txt).

## Results

### XQDT

The released scores include a parser correction that retains error rows containing
`type`. The original model responses and human annotations are unchanged; the
largest change in the nine correlations is 0.87 percentage points relative to
the previous parser. These corrected XQDT scores differ slightly from the
camera-ready tables.

The table below reports Spearman correlation ($\rho \times 100$) between model-predicted precision, recall, and F1 and the corresponding 4L-RP-Human scores.

| Model | $P_m$ vs $P_h$ | $R_m$ vs $R_h$ | $F1_m$ vs $F1_h$ |
| --- | ---: | ---: | ---: |
| Gemma3-270M | 75.7 | 83.5 | 82.8 |
| Gemma3-1B | 74.3 | 81.5 | 81.3 |
| Gemma3-4B | 79.4 | 88.0 | 87.8 |
| Gemma3-12B | 61.8 | 56.1 | 63.1 |
| Qwen3-0.6B | 71.8 | 79.3 | 78.4 |
| Qwen3-1.7B | 66.3 | 74.0 | 72.5 |
| Qwen3-4B | 88.8 | 87.3 | 88.4 |
| Qwen3-8B | 88.9 | 89.2 | 89.8 |
| Qwen3-14B | 60.5 | 65.1 | 69.8 |
| Llama3.2-1B | 71.6 | 77.7 | 77.3 |
| Llama3.2-3B | 85.0 | 80.4 | 85.9 |
| Llama3.1-8B | 82.6 | 78.6 | 84.5 |

Among the released XQDT models, Qwen3-8B gives the strongest overall correlations, reaching 88.9, 89.2, and 89.8 for precision, recall, and F1 respectively.

### Traditional baselines

The release also includes the 50 example-level outputs used to score DQE/QuestEval, MonoLR, NLI, and the original FactSpotter model.

| Baseline | P | R | F1 |
| --- | ---: | ---: | ---: |
| DQE / QuestEval | 61.2 | 58.3 | 62.5 |
| MonoLR | 71.0 | 63.7 | 70.6 |
| NLI | 75.6 | 76.6 | 77.5 |
| FactSpotter | --- | 86.5 | --- |

FactSpotter is recall-only. For DQE, the reported values are mean correlations over 1,000 bootstrap resamples with random seed 42.

## Quick Start

The saved predictions can be rescored directly without rerunning model inference.

```bash
python correlations/4L-RP-Human/score_4lrp_from_results.py \
  --results-dir correlations/4L-RP-Human/xqdt_results_repaired \
  --output-dir correlations/4L-RP-Human/xqdt_correlation_results

python correlations/4L-RP-Human/score_4lrp_from_results.py \
  --results-dir correlations/4L-RP-Human/prompted_llm_results \
  --output-dir correlations/4L-RP-Human/prompted_correlation_results

python correlations/4L-RP-Human/baselines/score_saved_baselines.py
```

The baseline command writes
`correlations/4L-RP-Human/baselines/results/rescored/baseline_correlations.json`.

To rebuild the parsed errors and sample-level scores from the saved raw responses:

```bash
python correlations/reparse_saved_xqdt.py --dataset 4lrp \
  --results-dir correlations/4L-RP-Human/xqdt_results_repaired \
  --output-dir /tmp/xqdt-4lrp-reparsed
python correlations/4L-RP-Human/score_4lrp_from_results.py \
  --results-dir /tmp/xqdt-4lrp-reparsed \
  --output-dir /tmp/xqdt-4lrp-correlations
```

## Annotation Data

The `by_language/` directory contains 50 evaluation examples for each of eight languages:

- Arabic
- Chinese
- English
- French
- Maltese
- Russian
- Spanish
- Welsh

The paper evaluates XQDT on the English subset. English, Maltese, Russian, and Welsh form the 4L-RP-Human benchmark described by [Soto Martinez et al. (2025)](https://aclanthology.org/2025.findings-acl.542/); Arabic, Chinese, French, and Spanish are additional annotation subsets released with this repository.

The same eight-language annotation release is available on [Hugging Face](https://huggingface.co/datasets/Loria-MosAIk/4L-RP-Human-Clean).

The de-identified source annotations are stored under:

```text
source_annotations/
```

They contain:

- `items.csv`: source texts, graphs, languages, levels, and question IDs;
- `annotations.csv`: individual precision and recall annotations;
- `annotators.csv`: repository-local anonymous annotator IDs and evaluation inclusion;
- `annotation_corrections.csv`: recorded annotation corrections applied during data preparation.

Participant identities and response times are not included.

Rebuild the processed language files with:

```bash
python correlations/4L-RP-Human/prepare_annotations.py
```

## Released Outputs

The main released inputs and cached predictions are:

- English annotations: `correlations/4L-RP-Human/by_language/english.json`
- XQDT outputs: `correlations/4L-RP-Human/xqdt_results_repaired/`
- prompted-LLM outputs: `correlations/4L-RP-Human/prompted_llm_results/`
- traditional baseline outputs: `correlations/4L-RP-Human/baselines/results/`

These files can be rescored directly without rerunning the corresponding models.

## Re-run XQDT Inference

The `llm_eval_*.py` scripts load the published WebNLG XQDT adapters from Hugging Face and evaluate them on the 4L-RP-Human examples.

```bash
python correlations/4L-RP-Human/llm_eval_gemma.py
USE_HF=1 python correlations/4L-RP-Human/llm_eval_qwen.py
python correlations/4L-RP-Human/llm_eval_llama.py
```

New predictions and correlations are written to:

```text
correlations/4L-RP-Human/xqdt_results/
```

Each run produces example-level predictions, per-model correlations, and an aggregate summary.

## Published XQDT Adapters

The 4L-RP-Human experiments use the WebNLG XQDT adapters:

- `Gemma3-270M` -> `Loria-MosAIk/xqdt-webnlg-gemma3-270m`
- `Gemma3-1B` -> `Loria-MosAIk/xqdt-webnlg-gemma3-1b`
- `Gemma3-4B` -> `Loria-MosAIk/xqdt-webnlg-gemma3-4b`
- `Gemma3-12B` -> `Loria-MosAIk/xqdt-webnlg-gemma3-12b`
- `Qwen3-0.6B` -> `Loria-MosAIk/xqdt-webnlg-qwen3-0.6b`
- `Qwen3-1.7B` -> `Loria-MosAIk/xqdt-webnlg-qwen3-1.7b`
- `Qwen3-4B` -> `Loria-MosAIk/xqdt-webnlg-qwen3-4b`
- `Qwen3-8B` -> `Loria-MosAIk/xqdt-webnlg-qwen3-8b`
- `Qwen3-14B` -> `Loria-MosAIk/xqdt-webnlg-qwen3-14b`
- `Llama3.2-1B` -> `Loria-MosAIk/xqdt-webnlg-llama3.2-1b`
- `Llama3.2-3B` -> `Loria-MosAIk/xqdt-webnlg-llama3.2-3b`
- `Llama3.1-8B` -> `Loria-MosAIk/xqdt-webnlg-llama3.1-8b`

## Prompted LLM Baselines

The four prompted baselines below are scored on all 50 English examples, with
no exclusions. Values are Spearman correlation ($\rho \times 100$) with the
corresponding human precision, recall, and F1 scores.

| Model | P | R | F1 |
| --- | ---: | ---: | ---: |
| GPT-5.1 | 80.9 | 76.8 | 78.5 |
| Gemma3-27B | 28.9 | 35.1 | 36.1 |
| Qwen3-32B | 35.9 | 30.4 | 35.0 |
| Llama3.3-70B-AWQ | 32.4 | 35.5 | 32.8 |

Each `*_results.json` file contains the input, prompt, raw model response, parsed errors, example-level precision/recall/F1, and human scores. Each `*_correlation.json` stores the corresponding correlations.

Use the prompted-results command in [Quick Start](#quick-start) to recompute these correlations.

To rerun prompted inference, inspect the available options with:

```bash
python correlations/4L-RP-Human/prompted_llm_4lrp.py --help
```

## Traditional Baselines

DQE / QuestEval, MonoLR, NLI, and FactSpotter each have 50 example-level
outputs under `correlations/4L-RP-Human/baselines/results/`. The scoring command
is included in [Quick Start](#quick-start).

MonoLR, NLI, and FactSpotter can be rerun with:

```bash
python correlations/4L-RP-Human/baselines/plm_baselines.py
```

FactSpotter uses the original checkpoint:

```text
checkpoints/fact_spotter_electra.pt
```

DQE / QuestEval can be rerun separately:

```bash
python -m pip install evaluate bert_score spacy sentencepiece unidecode
python -m spacy download en_core_web_sm
python correlations/4L-RP-Human/baselines/quest_eval_corr.py
```

## Environment

Model inference was prepared with:

- Python `3.11`
- `torch==2.9.1`
- the dependencies in `verifier_train_eval_webnlg/requirements-py311.txt`

A local environment can be created from the repository root with:

```bash
python3.11 -m venv .venv
./.venv/bin/python -m pip install -U pip
./.venv/bin/python -m pip install -r verifier_train_eval_webnlg/requirements-py311.txt
```
