# T2D Repair

This directory contains the inputs, saved repair outputs, and code for the text-to-data repair experiments in XQDT. The experiments use GenWiki and WebNLG, with BT5 and ReGen as the base extractors.

## Setup

Use Python 3.11 and run the commands below from the repository root.

### Scoring saved outputs

Install the evaluation dependencies and tokenizers:

```bash
python -m pip install -r repair_t2d/requirements-repair_t2d-eval-py311.txt
python -m nltk.downloader punkt punkt_tab
```

The evaluator uses `nervaluate==0.2.0` and `nltk==3.9.4`.

### Generating new repair outputs

For generation, install both dependency files: the launchers generate repaired triples and evaluate each completed round.

```bash
python -m pip install -r repair_t2d/requirements-repair_t2d-py311.txt \
  -r repair_t2d/requirements-repair_t2d-eval-py311.txt
python -m nltk.downloader punkt punkt_tab
```

## Inputs and Saved Outputs

- `t2d_repair_input/genwiki/` and `t2d_repair_input/webnlg/`: source texts, gold triples, and BT5/ReGen predictions.
- `eval/data/`: reference XML files for the Text-to-RDF evaluator.
- `results/repair_outputs/`: feedback and repaired triples for each method and round.
- `results/updated_scores.json`: saved precision, recall, and F1 scores.

Each dataset directory contains `<dataset>_test_unified.json`, `raw/<dataset>_bt5.json`, and `raw/<dataset>_regen.json`. Evaluation uses 1,000 GenWiki examples and 2,155 WebNLG examples.

The JSONL files under `results/repair_outputs/` store feedback and per-round changes to the triples:

| Method | Feedback | Repair output |
| --- | --- | --- |
| XQDT | `missing`, `extra`, and `incorrect` verifier feedback | repaired triples |
| FactSpotter | triples marked unsupported by FactSpotter | repaired triples |
| NLI | unsupported triples, a missing-fact flag, and a global NLI score | repaired triples |
| PiVe | predicted missing triples | repaired triples |
| Self-Refine | model-generated critique | repaired triples |

Reconstruction starts from the BT5/ReGen predictions and applies these records in round order. Samples absent from a round, or marked `fell_back=true`, keep their previous triples. File mappings and field definitions are in `results/repair_outputs/manifest.json`.

## Recompute the Main-Text and Appendix Results

Run the scorer on the saved triples, then export both sets of tables from the newly computed scores:

```bash
python repair_t2d/rescore_saved_outputs.py \
  --output-dir repair_t2d/results/rescored --workers 4

python repair_t2d/export_maintext_tables.py \
  --summary repair_t2d/results/rescored/scores.json

python repair_t2d/export_appendix_tables.py \
  --summary repair_t2d/results/rescored/scores.json
```

The scorer reconstructs the saved outputs and compares them with the reference XML files. It computes precision, recall, and F1 under the Ent_type, Partial, Strict, and Exact criteria for all five repair methods and nine repair models, on both datasets and base extractors, across R1/R2/R3.

Outputs:

- `repair_t2d/results/rescored/scores.json`: recomputed scores.
- `repair_t2d/exports/maintext_tables.md`: main-text Tables 9 and 10.
- `repair_t2d/exports/appendix_tables.md`: complete Exact, Strict, and Partial F1 tables for all nine repair models and all three rounds.

### Export the saved scores

To generate the same table layouts from `results/updated_scores.json` rather than recomputing scores, run:

```bash
python repair_t2d/export_maintext_tables.py
python repair_t2d/export_appendix_tables.py
```

## Results

The tables below report Exact F1 (%). BT5 and ReGen are the base extractors; G3, L3.1, and Q3 denote Gemma 3, Llama 3.1, and Qwen3 repair models. SR is Self-Refine, FS is FactSpotter, and R1/R3 denote the first/third repair round.

### Table 9 / GenWiki

| Method | BT5 G3-12B | BT5 L3.1-8B | BT5 Q3-14B | ReGen G3-12B | ReGen L3.1-8B | ReGen Q3-14B |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | 40.73 | 40.73 | 40.73 | 47.34 | 47.34 | 47.34 |
| XQDT-R1 | 48.58 | 44.65 | 43.19 | 50.57 | 49.59 | 48.32 |
| XQDT-R3 | **50.81** | **47.76** | **47.04** | **51.86** | **50.41** | **49.36** |
| SR-R1 | 42.13 | 40.25 | 41.46 | 44.06 | 47.11 | 44.17 |
| SR-R3 | 41.96 | 40.02 | 40.25 | 43.76 | 45.50 | 40.99 |
| FS-R1 | 37.73 | 40.13 | 33.67 | 44.56 | 46.49 | 43.41 |
| FS-R3 | 36.99 | 38.92 | 33.11 | 44.13 | 45.47 | 43.07 |
| NLI-R1 | 47.04 | 41.09 | 38.33 | 49.35 | 46.71 | 45.33 |
| NLI-R3 | 49.37 | 41.56 | 40.09 | 51.19 | 46.33 | 45.58 |
| PiVe-R1 | 40.73 | 40.66 | 40.73 | 47.34 | 47.25 | 47.34 |
| PiVe-R3 | 40.73 | 40.66 | 40.73 | 47.34 | 47.17 | 47.34 |

### Table 10 / WebNLG

| Method | BT5 G3-12B | BT5 L3.1-8B | BT5 Q3-14B | ReGen G3-12B | ReGen L3.1-8B | ReGen Q3-14B |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | 67.63 | 67.63 | 67.63 | 73.37 | 73.37 | 73.37 |
| XQDT-R1 | 72.22 | 69.76 | 72.53 | 75.18 | 74.39 | 75.51 |
| XQDT-R3 | **72.72** | **70.82** | **74.31** | **75.53** | **74.67** | **75.84** |
| SR-R1 | 58.53 | 62.22 | 62.27 | 61.07 | 66.26 | 65.74 |
| SR-R3 | 55.18 | 58.16 | 58.94 | 56.69 | 60.68 | 60.36 |
| FS-R1 | 65.46 | 67.38 | 64.83 | 73.06 | 73.59 | 73.23 |
| FS-R3 | 65.23 | 67.15 | 64.65 | 73.03 | 73.38 | 73.14 |
| NLI-R1 | 68.07 | 66.09 | 66.21 | 72.35 | 71.85 | 71.98 |
| NLI-R3 | 69.04 | 65.93 | 67.08 | 73.01 | 71.25 | 72.17 |
| PiVe-R1 | 67.63 | 67.52 | 67.63 | 73.37 | 73.23 | 73.37 |
| PiVe-R3 | 67.63 | 67.50 | 67.63 | 73.37 | 73.12 | 73.37 |

The appendix command above exports the full results, including round 2 and the additional repair models.

## Reconstruct and Score One Setting

For example, reconstruct the XQDT outputs from Gemma 3 12B on WebNLG with BT5:

```bash
python repair_t2d/reconstruct_compact_outputs.py \
  --method xqdt \
  --repair-model gemma3_12b \
  --dataset webnlg \
  --baseline bt5 \
  --output-dir repair_t2d/reconstructed/xqdt_gemma3_12b_webnlg_bt5
```

Available methods are `xqdt`, `factspotter`, `nli`, `pive`, and `self_refine`. The command writes evaluator-compatible JSON files under `round0/`, `round1/`, `round2/`, and `round3/`.

Score the reconstructed third round with:

```bash
python repair_t2d/eval/eval_text_to_data_repair.py \
  --repair-root repair_t2d/reconstructed/xqdt_gemma3_12b_webnlg_bt5/round3 \
  --datasets webnlg \
  --baselines bt5 \
  --repair-models gemma3_12b \
  --prompt-variants g_bridge_recover_guard
```

## Re-run Repair Generation

### XQDT

XQDT repair uses the published WebNLG Qwen3-8B verifier. Download its adapter once, then run the three repair rounds:

```bash
python verifier_train_eval_webnlg/download_verifier_adapters.py --dataset webnlg --model Qwen3-8B

USE_HF=1 bash repair_t2d/run_xqdt_round1.sh
USE_HF=1 bash repair_t2d/run_xqdt_round23.sh
```

Each round verifies the preceding output: R2 uses the R1-repaired triples, and R3 uses the R2-repaired triples.

### Repair baselines

Run PiVe, NLI, FactSpotter, and Self-Refine with:

```bash
bash repair_t2d/run_baseline_round1.sh
bash repair_t2d/run_baseline_round23.sh
```

Set `DATASETS`, `BASELINES`, and `REPAIR_MODELS` to select experiments, using the same selection for round 1 and rounds 2–3. Repair model IDs are defined in `run_text_to_data_verifier_repair.py`; the nine models in the result tables are listed in `results/repair_outputs/manifest.json`.
