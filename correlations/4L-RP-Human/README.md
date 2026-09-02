# 4L-RP-Human Correlation Workflow

The English annotations and cached XQDT and prompted-baseline outputs needed for Table 5 are bundled here. The XQDT adapters used for this evaluation are available in the [WebNLG model collection](https://huggingface.co/collections/Loria-MosAIk/xqdt-webnlg-models-6a94697a0582edd20ad3df04).

This README shows how to reproduce the XQDT correlation results reported for the 4L-RP-Human English subset. The release also includes eight language-specific annotation files.

## Tested Environment

This subtree was prepared for the same Python environment as the verifier workflows:

- Python `3.11`
- package versions listed in `verifier_train_eval_webnlg/requirements-py311.txt`
- `torch==2.9.1`

The commands below assume that you are already using a compatible Python 3.11 environment.
If you want a local virtual environment at the repository root, one explicit setup path is:

```bash
python3.11 -m venv .venv
./.venv/bin/python -m pip install -U pip
./.venv/bin/python -m pip install -r verifier_train_eval_webnlg/requirements-py311.txt
```

Install a PyTorch build that matches your local setup.
The original runs used `torch 2.9.1`.

## Annotation Data

The `by_language/` directory contains 50 evaluation examples for each of Arabic, Chinese, English, French, Maltese, Russian, Spanish, and Welsh. The XQDT paper reports results on English only. English, Maltese, Russian, and Welsh form the 4L-RP-Human benchmark described by [Soto Martinez et al. (2025)](https://aclanthology.org/2025.findings-acl.542/); the other four files are additional collected subsets and are not used in Table 5.

The de-identified source export before evaluation-subset selection is under `source_annotations/`:

- `items.csv`: source texts, graphs, languages, levels, and question IDs
- `annotations.csv`: individual precision and recall ratings
- `annotators.csv`: repository-local annotator IDs and evaluation inclusion
- `annotation_corrections.csv`: the recorded correction applied to the processed English input

Participant identifiers and response times are not included. Rebuild all eight language files with:

```bash
python correlations/4L-RP-Human/prepare_annotations.py
```

## Inputs and Outputs

The bundled paper results can be rescored directly from:

- English annotations: `correlations/4L-RP-Human/by_language/english.json`
- XQDT outputs: `correlations/4L-RP-Human/xqdt_results_repaired/`
- prompted-LLM outputs: `correlations/4L-RP-Human/prompted_llm_results/`

The XQDT inference scripts load their published adapters directly from Hugging Face and write new outputs to `correlations/4L-RP-Human/xqdt_results/`.

## 1. Score Saved XQDT Results

This is the fastest way to reproduce the bundled XQDT rows in the paper without running inference.

### Score the bundled paper results

```bash
python correlations/4L-RP-Human/score_4lrp_from_results.py \
  --results-dir correlations/4L-RP-Human/xqdt_results_repaired \
  --output-dir correlations/4L-RP-Human/xqdt_results_repaired
```

### Score a new inference run

```bash
python correlations/4L-RP-Human/score_4lrp_from_results.py \
  --results-dir correlations/4L-RP-Human/xqdt_results \
  --output-dir correlations/4L-RP-Human/xqdt_results
```

Outputs:

- per-model correlation JSON files: `*_correlation.json`
- aggregate summary: `summary.json`
- text log: `evaluation_YYYYMMDD_HHMMSS.log`

The default log path is inside `--output-dir`.

### Reproduced Table 5 scores

These are the reproduced XQDT rows from `xqdt_results_repaired/summary.json`. Values are Spearman correlations ($\rho \times 100$), rounded to one decimal place.

| Model | Pm vs Ph | Rm vs Rh | F1m vs F1h |
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

## 2. Re-run XQDT Inference

The `llm_eval_*.py` entry points download the published adapters from Hugging Face, compute the final P/R/F1 directly, and write final result files without a separate PRF repair step.

### Gemma adapters

```bash
python correlations/4L-RP-Human/llm_eval_gemma.py
```

### Qwen adapters

```bash
USE_HF=1 python correlations/4L-RP-Human/llm_eval_qwen.py
```

### Llama adapters

```bash
python correlations/4L-RP-Human/llm_eval_llama.py
```

Outputs are written to:

```text
correlations/4L-RP-Human/xqdt_results/
```

Each run writes:

- `*_results.json`
- `*_correlation.json`
- `summary.json`

## 3. Table 5 Adapters

The XQDT rows in Table 5 use the following published WebNLG adapters:

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

## 4. Baseline Scripts

The bundled prompted-LLM results cover GPT-5.1, Gemma 3 27B, Qwen3 32B, and Llama 3.3 70B. Each `*_results.json` file contains the input, prompt, raw response, parsed errors, example-level P/R/F1, and human scores; each `*_correlation.json` file contains the corresponding correlations. `prompted_llm_results/summary.json` aggregates the four models.

To inspect the options for rerunning a prompted baseline:

```bash
python correlations/4L-RP-Human/prompted_llm_4lrp.py --help
```

The baseline rows are reproduced separately from:

- `correlations/4L-RP-Human/baselines/plm_baselines.py`
- `correlations/4L-RP-Human/baselines/quest_eval_corr.py`

These baselines do not use the XQDT LoRA adapters listed above.

The default baseline wiring in this repository is:

- input annotations: `correlations/4L-RP-Human/by_language/english.json`
- NLI template file: `correlations/4L-RP-Human/baselines/webnlg_templates.json`
- FactSpotter-ELECTRA checkpoint: downloaded separately from the original implementation
- saved outputs: `correlations/4L-RP-Human/baselines/results/`

### Run the PLM baselines

```bash
python correlations/4L-RP-Human/baselines/plm_baselines.py
```

This writes:

- `monolr_results.json`
- `nli_results.json`
- `factspotter_electra_results.json`
- `factspotter_deberta_small_results.json`
- `factspotter_deberta_base_results.json`
- `factspotter_deberta_large_results.json`
- `plm_baselines_summary.json`
- `plm_baselines_evaluation_YYYYMMDD_HHMMSS.log`

### Run QuestEval

```bash
python correlations/4L-RP-Human/baselines/quest_eval_corr.py
```

This writes:

- `questeval_results.json`
- `questeval_correlation.json`
- `questeval_summary.json`
- `questeval_evaluation_YYYYMMDD_HHMMSS.log`

## 5. Sanity Checks

Check that the expected assets exist:

```bash
ls correlations/4L-RP-Human/by_language
ls correlations/4L-RP-Human/xqdt_results_repaired
ls correlations/4L-RP-Human/prompted_llm_results
```

Check the environment before running the scorer:

```bash
python -c "import torch, scipy; print(torch.__version__)"
```
