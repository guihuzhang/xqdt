# D2T Repair

This directory contains the inputs, saved repair outputs, and code for the data-to-text repair experiments in XQDT. The experiments use WebNLG 2020, with Control Prefix as the base generator.

## Results

Scores are percentages. G3 and L3.1 denote Gemma 3 and Llama 3.1 repair models; SR is Self-Refine and FS is FactSpotter. All methods start from the same Control Prefix outputs.

| Model | Method | BLEU | chrF++ | METEOR | PARENT-F1 | BERTScore-F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Control Prefix | Baseline | 53.81 | 69.18 | 41.58 | 65.60 | 95.72 |
| G3-4B | SR-R1 | 51.99 | 67.95 | 40.98 | 65.70 | 95.55 |
| G3-4B | SR-R3 | 51.54 | 67.62 | 40.82 | 65.65 | 95.51 |
| G3-4B | FS-R1 | 53.43 | 69.05 | 41.57 | 65.44 | 95.73 |
| G3-4B | FS-R3 | 53.35 | 69.11 | 41.63 | 65.50 | 95.72 |
| G3-4B | NLI-R1 | 52.67 | 68.38 | 41.20 | 65.09 | 95.64 |
| G3-4B | NLI-R3 | 52.48 | 68.38 | 41.22 | 65.18 | 95.64 |
| G3-4B | XQDT-R1 | 53.73 | 69.45 | 41.80 | 65.76 | 95.73 |
| G3-4B | XQDT-R3 | 53.69 | 69.49 | 41.85 | 65.81 | 95.73 |
| L3.1-8B | SR-R1 | 48.51 | 66.47 | 39.86 | 62.93 | 95.19 |
| L3.1-8B | SR-R3 | 47.26 | 65.98 | 39.72 | 62.65 | 95.09 |
| L3.1-8B | FS-R1 | 53.52 | 68.88 | 41.44 | 65.30 | 95.71 |
| L3.1-8B | FS-R3 | 53.45 | 69.21 | 41.65 | 65.21 | 95.72 |
| L3.1-8B | NLI-R1 | 53.05 | 68.77 | 41.38 | 65.00 | 95.71 |
| L3.1-8B | NLI-R3 | 53.08 | 69.02 | 41.60 | 65.17 | 95.72 |
| L3.1-8B | XQDT-R1 | 53.93 | 69.48 | 41.82 | 65.73 | 95.75 |
| L3.1-8B | XQDT-R3 | 53.88 | 69.55 | 41.86 | 65.71 | 95.75 |

## Setup

Use Python 3.11 and run the commands below from the repository root.

### Scoring saved outputs

Install the evaluation dependencies and tokenizer data:

```bash
python -m pip install -r repair_d2t/requirements-repair_d2t-eval-py311.txt
python -m nltk.downloader punkt_tab
```

For METEOR, install Java and download the complete METEOR 1.5 distribution from its [official download page](https://www.cs.cmu.edu/~alavie/METEOR/). Extract it, including its data files, under `repair_d2t/metrics/meteor-1.5/`. The JAR should be at `repair_d2t/metrics/meteor-1.5/meteor-1.5.jar`.

The chrF++ implementation and PARENT code are included under `metrics/`. PARENT uses NLTK 3.9.4. BERTScore and SBERT download their scoring models on first use; SBERT uses `sentence-transformers/all-mpnet-base-v2`.

### Generating new repair outputs

Install the generation dependencies, which also include the evaluation packages:

```bash
python -m pip install -r repair_d2t/requirements-repair_d2t-py311.txt
python -m nltk.downloader punkt_tab
```

The launchers generate three repair rounds and then score the outputs using the resources above.

## Inputs and Saved Outputs

- `inputs/webnlg20_control_prefix_20_eval_input.json`: 1,779 Control Prefix outputs, English references, normalized source triples, and raw source triples.
- `results/repair_outputs/`: per-round feedback and repaired texts for four methods and nine repair models.
- `results/repair_outputs/manifest.json`: model IDs, experiment settings, and field definitions.
- `repair/`: feedback generation, repair prompts, and model inference.
- `scoring/`: text evaluation and round-by-round scoring.

The saved JSONL files contain:

| Method | Feedback | Repair output |
| --- | --- | --- |
| XQDT | missing, extra, and incorrect units identified by the verifier | repaired text and model response |
| FactSpotter | facts marked unsupported by FactSpotter | repaired text |
| NLI | unsupported facts and an unsupported-content warning | repaired text |
| Self-Refine | model-generated critique | critique and revised text |

Each record's `index` refers to an example in the shared input. Reconstruction starts from the Control Prefix texts and applies the records in round order. Examples without a record in a round keep their previous text.

## Recompute the Main-Text and Appendix Results

Reconstruct all saved texts and score all four methods, nine repair models, and three rounds:

```bash
python repair_d2t/rescore_saved_outputs.py \
  --output-dir repair_d2t/results/rescored
```

The scorer compares each round's texts with the references and source triples. It computes BLEU, chrF++, METEOR, PARENT-F1, BERTScore-F1, ROUGE-L, and SBERT.

Outputs:

- `repair_d2t/results/rescored/scores.json`: recomputed scores for each method, model, and round.
- `repair_d2t/results/rescored/maintext_tables.md`: the main-text comparison for Gemma 3 4B and Llama 3.1 8B, using R1 and R3.
- `repair_d2t/results/rescored/appendix_tables.md`: all nine repair models, all seven metrics, and R1/R2/R3.
- `repair_d2t/results/rescored/reconstructed/`: complete texts for the baseline and each repair round.

To recompute selected metrics or settings, use `--metrics`, `--methods`, and `--model-tags`. For example:

```bash
python repair_d2t/rescore_saved_outputs.py \
  --methods xqdt \
  --model-tags gemma3_4b_it llama3_1_8b_instruct \
  --metrics bleu,chrf++ \
  --output-dir repair_d2t/results/rescored_bleu_chrf
```

## Reconstruct and Score One Setting

For example, reconstruct the XQDT outputs from Qwen3-8B:

```bash
python repair_d2t/reconstruct_compact_outputs.py \
  --method xqdt \
  --model-tag qwen3_8b_nothink \
  --output-dir repair_d2t/reconstructed/qwen3_8b_xqdt
```

This writes `round0_baseline.json` and `round1_improved.json` through `round3_improved.json`. Score these texts with:

```bash
python repair_d2t/scoring/score_improve_rounds_full.py \
  --run_dir repair_d2t/reconstructed/qwen3_8b_xqdt \
  --reference_json repair_d2t/inputs/webnlg20_control_prefix_20_eval_input.json \
  --metrics bleu,chrf++,meteor,parent,bertscore,rouge_l,sbert
```

The scores are written to `text_metrics/round_metrics_full.json` inside the reconstructed directory.

## Re-run Repair Generation

All repair methods use temperature 0.7 and seed 42. XQDT uses the published WebNLG Qwen3-8B verifier with judge temperature 0.3, judge seed 2023, and the `p1c_p2b` repair prompt in `repair/prompt_variants_v1.py`.

### XQDT

```bash
USE_HF=1 bash repair_d2t/run_xqdt_real_baselines_multimodel_r3.sh
```

### Self-Refine

```bash
bash repair_d2t/run_self_refine_real_baselines_multimodel_r3.sh
```

Self-Refine uses the same `markdown_table` critique-and-revision prompt for all nine repair models.

### NLI and FactSpotter

```bash
FEEDBACK_MODE=nli \
bash repair_d2t/run_signal_feedback_real_baselines_multimodel_r3.sh

FEEDBACK_MODE=factspotter \
bash repair_d2t/run_signal_feedback_real_baselines_multimodel_r3.sh
```

NLI uses `roberta-large-mnli`, the WebNLG relation templates in `metrics/nli_dusek/webnlg_templates.json`, and an entailment threshold of 0.5. FactSpotter uses `Inria-CEDAR/FactSpotter-DeBERTaV3-Base` with a support threshold of 0.5.

The launchers run all nine repair models by default. Set `MODEL_TAGS` to select models and `OUT_ROOT` to choose an output directory. For example:

```bash
USE_HF=1 MODEL_TAGS=qwen3_8b_nothink OUT_ROOT=xqdt_outputs/qwen3_8b \
bash repair_d2t/run_xqdt_real_baselines_multimodel_r3.sh
```

Model IDs and tensor-parallel settings are defined in `repair_qwen8_fixed_judge_common.sh`. The launchers accept `GPU`, `QWEN14_TP`, `QWEN32_TP`, `GEMMA12_TP`, `GEMMA27_TP`, and `LLAMA70_TP` to select devices and tensor parallelism. Set `METRICS=bleu,chrf++` to score only these two metrics after generation.

| Model tag | Model |
| --- | --- |
| `gemma3_4b_it` | `google/gemma-3-4b-it` |
| `gemma3_12b_it` | `google/gemma-3-12b-it` |
| `gemma3_27b_it` | `google/gemma-3-27b-it` |
| `llama3_1_8b_instruct` | `meta-llama/Llama-3.1-8B-Instruct` |
| `llama3_3_70b_awq` | `casperhansen/llama-3.3-70b-instruct-awq` |
| `qwen3_4b_nothink` | `Qwen/Qwen3-4B` |
| `qwen3_8b_nothink` | `Qwen/Qwen3-8B` |
| `qwen3_14b_nothink` | `Qwen/Qwen3-14B` |
| `qwen3_32b_nothink` | `Qwen/Qwen3-32B` |

## Metric Implementations

BLEU is the mean sentence-level SacreBLEU score with effective order enabled. chrF++ uses the included `metrics/chrF.py` implementation with character order 6, word order 2, and beta 2. PARENT evaluates the texts against both the references and source triples.

Third-party license notices for chrF++ and PARENT are in `metrics/THIRD_PARTY_NOTICES.md`.
