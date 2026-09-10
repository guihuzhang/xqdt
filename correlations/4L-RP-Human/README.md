# 4L-RP-Human Correlation Workflow

This directory contains the human annotations, example-level model predictions, and evaluation code used for the 4L-RP-Human experiments in XQDT.

4L-RP-Human provides human precision and recall judgements for data-text pairs, allowing us to directly measure how well automatic metrics agree with human assessments of data-text alignment. The paper reports results on the English subset, while this release also includes processed annotations for eight languages.

Cached outputs are provided for XQDT, prompted LLM baselines, and four traditional baselines. The XQDT adapters are available in the [WebNLG model collection](https://huggingface.co/collections/Loria-MosAIk/xqdt-webnlg-models-6a94697a0582edd20ad3df04).

## Results

### XQDT

The table below reports Spearman correlation ($\rho \times 100$) between model-predicted precision, recall, and F1 and the corresponding 4L-RP-Human scores.

| Model | $P_m$ vs $P_h$ | $R_m$ vs $R_h$ | $F1_m$ vs $F1_h$ |
| --- | ---: | ---: | ---: |
| Gemma3-270M | 75.7 | 83.5 | 82.8 |
| Gemma3-1B | 74.3 | 81.5 | 81.3 |
| Gemma3-4B | 79.0 | 87.6 | 87.4 |
| Gemma3-12B | 61.5 | 55.7 | 62.7 |
| Qwen3-0.6B | 71.8 | 79.3 | 78.4 |
| Qwen3-1.7B | 66.3 | 74.0 | 72.5 |
| Qwen3-4B | 88.5 | 86.9 | 88.1 |
| Qwen3-8B | 88.6 | 88.9 | 89.6 |
| Qwen3-14B | 60.2 | 64.8 | 69.5 |
| Llama3.2-1B | 71.0 | 77.0 | 76.7 |
| Llama3.2-3B | 85.0 | 80.4 | 85.9 |
| Llama3.1-8B | 82.4 | 78.3 | 84.3 |

Among the released XQDT models, Qwen3-8B gives the strongest overall correlations, reaching 88.6, 88.9, and 89.6 for precision, recall, and F1 respectively.

### Traditional baselines

The release also includes the 50 example-level outputs used to score DQE/QuestEval, MonoLR, NLI, and the original FactSpotter-ELECTRA model.

| Baseline | P | R | F1 |
| --- | ---: | ---: | ---: |
| DQE / QuestEval | 61.2 | 58.3 | 62.5 |
| MonoLR | 71.0 | 63.7 | 70.6 |
| NLI | 75.6 | 76.6 | 77.5 |
| FactSpotter-ELECTRA | --- | 86.5 | --- |

FactSpotter is recall-only. For DQE, the reported values are mean correlations over 1,000 bootstrap resamples with random seed 42.

## Quick Start

The saved predictions can be rescored directly without rerunning model inference.

From the repository root:

```bash
python correlations/4L-RP-Human/score_4lrp_from_results.py \
  --results-dir correlations/4L-RP-Human/xqdt_results_repaired \
  --output-dir correlations/4L-RP-Human/xqdt_correlation_results

python correlations/4L-RP-Human/baselines/score_saved_baselines.py
```

The first command recomputes correlations for the saved XQDT outputs. The second recomputes correlations for the four traditional baselines and writes:

```text
correlations/4L-RP-Human/baselines/results/rescored/baseline_correlations.json
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

### Gemma

```bash
python correlations/4L-RP-Human/llm_eval_gemma.py
```

### Qwen

```bash
USE_HF=1 python correlations/4L-RP-Human/llm_eval_qwen.py
```

### Llama

```bash
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

The complete collection is available at:

https://huggingface.co/collections/Loria-MosAIk/xqdt-webnlg-models-6a94697a0582edd20ad3df04

## Prompted LLM Baselines

The released prompted baselines are:

- GPT-5.1
- Gemma 3 27B
- Qwen3 32B
- Llama 3.3 70B

Each `*_results.json` file contains the input, prompt, raw model response, parsed errors, example-level precision/recall/F1, and human scores. Each `*_correlation.json` stores the corresponding correlations.

Recompute correlations from the saved outputs with:

```bash
python correlations/4L-RP-Human/score_4lrp_from_results.py \
  --results-dir correlations/4L-RP-Human/prompted_llm_results \
  --output-dir correlations/4L-RP-Human/prompted_correlation_results
```

To rerun prompted inference, inspect the available options with:

```bash
python correlations/4L-RP-Human/prompted_llm_4lrp.py --help
```

## Traditional Baselines

The English evaluation uses four traditional metrics:

- **DQE / QuestEval**
- **MonoLR**
- **NLI**
- **FactSpotter-ELECTRA**

All four have 50 example-level outputs under:

```text
correlations/4L-RP-Human/baselines/results/
```

Recompute their correlations together with:

```bash
python correlations/4L-RP-Human/baselines/score_saved_baselines.py
```

MonoLR, NLI, and FactSpotter can be rerun with:

```bash
python correlations/4L-RP-Human/baselines/plm_baselines.py
```

FactSpotter uses the original ELECTRA checkpoint:

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
