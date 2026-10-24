# E2E Verifier

This directory contains the E2E synthetic data, predictions from 12 fine-tuned XQDT verifiers and five prompted baselines, and the code for training, inference, and evaluation. The E2E verifiers are trained on the combined WebNLG and E2E synthetic data. Published adapters are available in the [E2E model collection](https://huggingface.co/collections/Loria-MosAIk/xqdt-e2e-models-6a94697d5be8186d7795da71).

Run the commands below from the repository root.

## Strict F1

Scores below are recomputed from the saved predictions against the corrected test annotations and rounded to one decimal place.

| Model | Extra F1 | Omitted F1 | Incorrect F1 | Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| GPT-5.1-Prompt | 35.7 | 82.4 | 78.7 | 65.6 |
| GPT-4.1-Prompt | 33.1 | 41.2 | 62.1 | 45.5 |
| G3-27B-Prompt | 4.8 | 16.4 | 36.6 | 19.3 |
| Q3-32B-Prompt | 6.6 | 1.7 | 43.3 | 17.2 |
| L3.3-70B-Prompt | 19.8 | 57.1 | 54.3 | 43.7 |
| G3-270M | 92.9 | 94.6 | 95.6 | 94.4 |
| G3-1B | 93.5 | 97.3 | 96.7 | 95.8 |
| G3-4B | 97.1 | 99.0 | 97.9 | 98.0 |
| G3-12B | 97.7 | 99.3 | 97.8 | 98.3 |
| Q3-0.6B | 96.3 | 98.7 | 97.7 | 97.6 |
| Q3-1.7B | 96.5 | 99.0 | 97.7 | 97.7 |
| Q3-4B | 97.7 | 99.1 | 97.9 | 98.2 |
| Q3-8B | 97.5 | 99.0 | 97.7 | 98.1 |
| Q3-14B | 97.9 | 99.2 | 97.9 | 98.3 |
| L3.2-1B | 93.9 | 97.0 | 96.3 | 95.7 |
| L3.2-3B | 97.1 | 98.1 | 97.1 | 97.4 |
| L3.1-8B | 97.8 | 98.0 | 97.3 | 97.7 |

## Environment

With Python 3.11, install the scoring dependencies:

```bash
python -m pip install -r correlations/requirements.txt
```

For GPU training and inference, use the shared [WebNLG verifier environment](../verifier_train_eval_webnlg/README.md#run-xqdt-inference).

## Contents

- `data/`: corrected joint WebNLG+E2E training and development data, and the E2E test set.
- `xqdt_outputs/`: predictions from the 12 fine-tuned verifiers.
- `prompt_baselines/`: predictions from the five prompted baselines, including inputs, prompts, raw responses, and parsed errors.
- `verifier_eval/` and `verifier_train/`: XQDT inference and training scripts.

## Recompute Scores

```bash
python verifier_train_eval_webnlg/score_verifier_release.py --dataset e2e19 --output-dir verifier_train_eval_e2e19/evaluation_results
```

The shared scorer evaluates 1,229 examples per model using strict and element matching. It writes full-precision metrics to JSON and a strict-F1 table to Markdown.

For BGE-large-en-v1.5 similarity matching, install Sentence Transformers and add `--similarity`:

```bash
python -m pip install sentence-transformers==5.4.1
python verifier_train_eval_webnlg/score_verifier_release.py --dataset e2e19 --similarity --output-dir verifier_train_eval_e2e19/evaluation_results/similarity
```

The released data include annotation corrections. The published adapters were trained on the earlier data; training with this release uses the corrected version. Details are in [data/README.md](data/README.md).

## Run XQDT Inference

The scripts download the published E2E adapters from Hugging Face. To run Qwen3-4B:

```bash
USE_HF=1 python verifier_train_eval_e2e19/verifier_eval/swift_inference_qwen_e2e.py --model Qwen3-4B
```

Omit `--model` to run a full model family:

```bash
python verifier_train_eval_e2e19/verifier_eval/swift_inference_gemma_e2e.py
USE_HF=1 python verifier_train_eval_e2e19/verifier_eval/swift_inference_qwen_e2e.py
python verifier_train_eval_e2e19/verifier_eval/swift_inference_llama_e2e.py
```

All three scripts accept `--model`, `--max-samples`, and `--output-dir`. Predictions are saved under `verifier_train_eval_e2e19/xqdt_outputs/` by default.

## Train XQDT

Training uses the joint WebNLG+E2E files in `data/` and the spaCy English pipeline:

```bash
python -m spacy download en_core_web_sm
```

### Gemma

```bash
python verifier_train_eval_e2e19/verifier_train/swift_simple_train_gemma3_270e2e.py
python verifier_train_eval_e2e19/verifier_train/swift_simple_train_gemma3_1b_e2e.py
python verifier_train_eval_e2e19/verifier_train/swift_simple_train_gemma3_4b_e2e.py
python verifier_train_eval_e2e19/verifier_train/swift_simple_train_gemma3_12b_e2e.py
```

### Qwen

```bash
USE_HF=1 python verifier_train_eval_e2e19/verifier_train/swift_simple_train_qwen3_0.6e2e.py
USE_HF=1 python verifier_train_eval_e2e19/verifier_train/swift_simple_train_qwen3_1.7e2e.py
USE_HF=1 python verifier_train_eval_e2e19/verifier_train/swift_simple_train_qwen3_4b_e2e.py
USE_HF=1 python verifier_train_eval_e2e19/verifier_train/swift_simple_train_qwen3_8b_e2e.py
USE_HF=1 python verifier_train_eval_e2e19/verifier_train/swift_simple_train_qwen3_14b_e2e.py
```

### Llama

```bash
python verifier_train_eval_e2e19/verifier_train/swift_simple_train_llama3.2_1b_e2e.py
python verifier_train_eval_e2e19/verifier_train/swift_simple_train_llama3.2_3b_e2e.py
python verifier_train_eval_e2e19/verifier_train/swift_simple_train_llama3.1_8b_e2e.py
```
