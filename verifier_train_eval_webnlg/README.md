# WebNLG Verifier

This directory contains the WebNLG synthetic data, predictions from 12 fine-tuned XQDT verifiers and five prompted baselines, and the code for training, inference, and evaluation. The published adapters are available in the [WebNLG model collection](https://huggingface.co/collections/Loria-MosAIk/xqdt-webnlg-models-6a94697a0582edd20ad3df04).

Run the commands below from the repository root.

## Contents

- `data/`: corrected training, development, and test data, plus the synthetic-data audit.
- `xqdt_outputs/`: predictions from the 12 fine-tuned verifiers.
- `prompt_baselines/`: predictions from the five prompted baselines, including inputs, prompts, raw responses, and parsed errors.
- `score_verifier_release.py`: scores both groups against the corrected test annotations.
- `verifier_eval/` and `verifier_train/`: XQDT inference and training scripts.

## Recompute Scores

With Python 3.11, install the scoring dependencies and run:

```bash
python -m pip install -r correlations/requirements.txt
python verifier_train_eval_webnlg/score_verifier_release.py --dataset webnlg --output-dir verifier_train_eval_webnlg/evaluation_results
```

The scorer evaluates 1,779 examples per model using strict and element matching. It writes full-precision metrics to JSON and a strict-F1 table to Markdown.

For similarity matching, install Sentence Transformers and add `--similarity`. This uses BGE-large-en-v1.5 embeddings with maximum-similarity one-to-one matching:

```bash
python -m pip install sentence-transformers==5.4.1
python verifier_train_eval_webnlg/score_verifier_release.py --dataset webnlg --similarity --output-dir verifier_train_eval_webnlg/evaluation_results/similarity
```

The released data include annotation corrections. The published adapters were trained on the earlier data; training with this release uses the corrected version. Details are in [data/README.md](data/README.md).

### Strict F1

Scores below are recomputed from the saved predictions against the corrected test annotations and rounded to one decimal place.

| Model | Extra F1 | Omitted F1 | Incorrect F1 | Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| Gemma3-270M | 57.1 | 91.9 | 87.7 | 78.9 |
| Gemma3-1B | 66.8 | 94.5 | 92.3 | 84.5 |
| Gemma3-4B | 78.3 | 97.6 | 96.5 | 90.8 |
| Gemma3-12B | 81.3 | 96.5 | 97.0 | 91.6 |
| Qwen3-0.6B | 69.7 | 96.2 | 95.2 | 87.1 |
| Qwen3-1.7B | 73.8 | 96.9 | 96.1 | 88.9 |
| Qwen3-4B | 77.5 | 97.2 | 96.6 | 90.4 |
| Qwen3-8B | 79.2 | 96.8 | 96.9 | 90.9 |
| Qwen3-14B | 81.3 | 96.8 | 97.1 | 91.7 |
| Llama3.2-1B | 71.6 | 92.4 | 93.1 | 85.7 |
| Llama3.2-3B | 78.6 | 93.7 | 95.0 | 89.1 |
| Llama3.1-8B | 82.2 | 93.8 | 95.6 | 90.5 |
| GPT-4.1-Prompt | 16.8 | 30.8 | 64.1 | 37.2 |
| GPT-5.1-Prompt | 18.0 | 81.7 | 78.1 | 59.2 |
| Gemma3-27B-Prompt | 4.6 | 14.4 | 43.2 | 20.7 |
| Qwen3-32B-Prompt | 2.8 | 1.2 | 47.6 | 17.2 |
| Llama3.3-70B-Prompt | 13.2 | 51.8 | 52.3 | 39.1 |

## Run XQDT Inference

Training and inference use Python 3.11 and the shared verifier dependencies. Install a PyTorch build suitable for your GPU; the original experiments used PyTorch 2.9.1.

```bash
python -m pip install -r verifier_train_eval_webnlg/requirements-py311.txt
```

Download the published Qwen3-4B adapter, then run it on the test set:

```bash
python verifier_train_eval_webnlg/download_verifier_adapters.py --dataset webnlg --model Qwen3-4B
USE_HF=1 python verifier_train_eval_webnlg/verifier_eval/swift_inference_qwen.py --model Qwen3-4B
```

Omit `--model` from the download command to download all 12 adapters. To run the model groups:

```bash
python verifier_train_eval_webnlg/verifier_eval/swift_inference_gemma.py
USE_HF=1 python verifier_train_eval_webnlg/verifier_eval/swift_inference_qwen.py
USE_HF=1 python verifier_train_eval_webnlg/verifier_eval/swift_inference_big.py
python verifier_train_eval_webnlg/verifier_eval/swift_inference_llama.py
```

All four scripts accept `--model`, `--max-samples`, and `--output-dir`. The `big` script runs Qwen3-8B, Qwen3-14B, and Llama3.1-8B; the other scripts run Gemma or the smaller Qwen and Llama models. Predictions are saved under `verifier_train_eval_webnlg/xqdt_outputs/` by default.

## Train XQDT

Training also uses the spaCy English pipeline:

```bash
python -m spacy download en_core_web_sm
```

### Gemma

```bash
python verifier_train_eval_webnlg/verifier_train/swift_simple_train_gemma3_270correct.py
python verifier_train_eval_webnlg/verifier_train/swift_simple_train_gemma3_1b_correct.py
python verifier_train_eval_webnlg/verifier_train/swift_simple_train_gemma3_4b_correct.py
python verifier_train_eval_webnlg/verifier_train/swift_simple_train_gemma3_12b_correct.py
```

### Qwen

```bash
USE_HF=1 python verifier_train_eval_webnlg/verifier_train/swift_simple_train_qwen3_0.6correct.py
USE_HF=1 python verifier_train_eval_webnlg/verifier_train/swift_simple_train_qwen3_1.7correct.py
USE_HF=1 python verifier_train_eval_webnlg/verifier_train/swift_simple_train_qwen3_4b_correct.py
USE_HF=1 python verifier_train_eval_webnlg/verifier_train/swift_simple_train_qwen3_8b_correct.py
USE_HF=1 python verifier_train_eval_webnlg/verifier_train/swift_simple_train_qwen3_14b_correct.py
```

### Llama

```bash
python verifier_train_eval_webnlg/verifier_train/swift_simple_train_llama3.2_1b_correct.py
python verifier_train_eval_webnlg/verifier_train/swift_simple_train_llama3.2_3b_correct.py
python verifier_train_eval_webnlg/verifier_train/swift_simple_train_llama3.1_8b_correct.py
```
