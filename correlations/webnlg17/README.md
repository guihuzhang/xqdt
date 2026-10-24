# WebNLG 2017 Human-Judgment Correlations

This directory contains the official WebNLG 2017 human-evaluation sample,
cached example-level metric scores, and the scoring code used for the WebNLG
2017 results in XQDT. The evaluation in Appendix Table 18 correlates
model-predicted F1 with human ratings of **semantic
adequacy** at both system and text levels across 223 inputs from 9 systems.

## Results

Correlations are multiplied by 100 and rounded to one decimal place.

| Model | System r | System rho | System tau | Text r | Text rho | Text tau |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BLEU | 77.0 | 71.6 | 56.8 | 77.6 | 79.6 | 67.0 |
| METEOR | 86.8 | 83.5 | 67.8 | 79.9 | 80.2 | 69.3 |
| PARENT | 88.2 | 82.6 | 66.7 | 79.2 | 79.6 | 66.9 |
| BERTScore | 70.1 | 78.3 | 63.7 | 80.0 | 81.1 | 69.5 |
| BARTScore | 91.0 | 87.5 | 78.2 | 81.4 | 81.8 | 70.9 |
| BLEURT | 90.5 | 88.1 | 72.1 | 82.1 | 82.6 | 71.4 |
| Data-QuestEval-F1 | 94.9 | 93.4 | 85.9 | 78.1 | 79.3 | 67.6 |
| FactSpotter | 97.2 | 93.9 | 85.1 | 84.7 | 81.8 | 70.3 |
| NLI-F1 | 94.5 | 95.0 | 87.0 | 82.6 | 78.5 | 68.3 |
| MonoLR-F1 | 95.1 | 89.8 | 77.9 | 84.1 | 81.4 | 70.4 |
| Gemma3-270M-F1 | 94.8 | 88.7 | 79.6 | 84.0 | 82.4 | 75.2 |
| Gemma3-1B-F1 | 93.0 | 86.8 | 76.8 | 83.5 | 82.2 | 74.7 |
| Gemma3-4B-F1 | 93.2 | 87.7 | 78.2 | 82.7 | 82.2 | 74.9 |
| Gemma3-12B-F1 | 91.6 | 87.9 | 78.2 | 83.6 | 82.5 | 75.1 |
| Qwen3-0.6B-F1 | 92.9 | 87.9 | 78.3 | 83.3 | 83.0 | 75.9 |
| Qwen3-1.7B-F1 | 92.5 | 87.1 | 77.4 | 83.6 | 83.0 | 75.4 |
| Qwen3-4B-F1 | 91.4 | 85.3 | 74.4 | 82.2 | 82.5 | 75.2 |
| Qwen3-8B-F1 | 91.2 | 85.4 | 73.6 | 82.4 | 82.4 | 75.2 |
| Qwen3-14B-F1 | -- | 67.6 | 56.0 | 81.2 | 79.1 | 71.7 |
| Llama3.2-1B-F1 | 91.8 | 87.0 | 76.5 | 83.6 | 82.5 | 75.4 |
| Llama3.2-3B-F1 | 92.1 | 87.1 | 77.2 | 83.0 | 82.8 | 75.5 |
| Llama3.1-8B-F1 | 92.2 | 86.6 | 76.3 | 82.3 | 81.6 | 74.8 |

## Environment

With Python 3.11, install the scoring dependencies from the repository root:

```bash
python -m pip install -r correlations/requirements.txt
```

## Contents

- `webnlg2017_inputs/`: the official human-evaluation sample, with input
  triples in `triples.txt` and system outputs in `teams-generations/*.txt`.
- `human_eval_webnlg_17.json`: triples, references, system outputs, and
  released human ratings in one benchmark file.
- `human-annotations/semantics/`: semantic-adequacy scores arranged by system.
- `metric_results/`: cached example-level scores for the metrics in Appendix
  Table 18.
- `xqdt_results/`: the 12 XQDT models' saved prompts, raw responses, and
  original parsed error lists (2,007 examples per model).
- `score_webnlg17.py`: standalone correlation reproduction script.
- `legacy_logs/`: full-precision logs from the original experiment runs.

## Recompute Correlations

Recompute correlations from the bundled example-level metric scores:

```bash
python correlations/webnlg17/score_webnlg17.py
```

To rebuild the XQDT score files from the original responses before scoring:

```bash
python correlations/webnlg17/score_webnlg17.py --rebuild-xqdt
```

This refreshes the XQDT F1 files in `metric_results/`, then computes correlations.

The default settings are:

- 1,000 paired bootstrap samples with random seed 10
- Pearson's r, Spearman's rho, and Kendall's tau
- system-level and text-level correlations

Results are saved as JSON and a Markdown table under
`correlations/webnlg17/results/`; use `--output-dir` to write them elsewhere.

## Evaluation Protocol

XQDT parser corrections change 298 of the 24,084 example-level F1 scores.
The largest correlation change is 2.5 points for Qwen3-14B.
F1 is rebuilt from error counts as `2TP / (2TP + FP + FN)`.

For each metric, the scorer pairs a `9 x 223` matrix of metric scores with the
matching matrix of human semantic-adequacy ratings.

**System level.** Each bootstrap replicate resamples the inputs with
replacement, averages the scores within each system, and computes the
correlation across systems.

**Text level.** For each input, the correlation is computed across the system
outputs; the retained per-input correlations are then averaged over the
bootstrap replicates.

Following the evaluation protocol used in the [FactSpotter
paper](https://aclanthology.org/2023.findings-emnlp.672/), correlations with
`p < 0.05` are retained within each bootstrap replicate.

## Original WebNLG 2017 Human-Evaluation Inputs

`webnlg2017_inputs/` is copied verbatim from the official
[WebNLG 2017 human-evaluation release](https://gitlab.com/webnlg/webnlg-human-evaluation):

```text
webnlg2017_inputs/
├── triples.txt
└── teams-generations/
    ├── adapt.txt
    ├── baseline.txt
    ├── melbourne.txt
    ├── pkuwriter.txt
    ├── tilburg-nmt.txt
    ├── tilburg-pipe.txt
    ├── tilburg-smt.txt
    ├── upf-forge.txt
    └── vietnam.txt
```

`triples.txt` is the organizers' `MRs.txt`: one triple set per line, with
`<br>` separating multiple triples.

## Cached Metric Scores

The saved model responses come from the original
[experiment repository](https://gitlab.inria.fr/mosaik/dt-explainable-metric/-/tree/18ee56fd1cf5d45bb9c662d6fd04fc9e90bd3d43/data/webnlg_annotation/datasets/webnlg2017/evaluation_results).
`--rebuild-xqdt` reparses these responses to compute F1.
The parser retains error rows whose triples contain `type`, handles explicit
`All incorrect` and `All rows are missing` responses, and reads multiline
table cells.
