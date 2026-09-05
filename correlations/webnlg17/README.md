# WebNLG 2017 Human-Judgment Correlations

This directory contains the official WebNLG 2017 human-evaluation sample,
cached example-level metric scores, and the scoring code used for the WebNLG
2017 results in XQDT. It reproduces Appendix Table 18 of the camera-ready
paper, which correlates model-predicted F1 with human ratings of **semantic
adequacy** at both system and text levels across 223 inputs from 9 systems.

## Environment

Requires Python 3.11 with the pins from
`verifier_train_eval_webnlg/requirements-py311.txt`:

- `numpy==2.3.5`
- `scipy==1.17.1`

Install just these two for the correlation reproduction:

```bash
pip install numpy==2.3.5 scipy==1.17.1
```

## Contents

- `webnlg2017_inputs/`: the official human-evaluation sample — `triples.txt`
  and `teams-generations/*.txt` per system.
- `human_eval_webnlg_17.json`: triples, references, system outputs, and
  released human ratings in one benchmark file.
- `human-annotations/semantics/`: semantic-adequacy scores arranged by system.
- `metric_results/`: cached example-level scores for the metrics in Appendix
  Table 18.
- `score_webnlg17.py`: standalone correlation reproduction script.
- `legacy_logs/`: full-precision logs from the original experiment runs.

## Reproduce Appendix Table 18

The fastest path starts from the bundled example-level metric scores; no
model inference is required.

```bash
python correlations/webnlg17/score_webnlg17.py
```

The scorer enumerates the 22 rows of Appendix Table 18 from its own
`WEBLG17_METRICS` constant, so it needs no external index. It uses the paper
settings by default:

- 1,000 paired bootstrap samples with random seed 10
- Pearson's r, Spearman's rho, and Kendall's tau
- system-level and text-level correlations

It writes a machine-readable JSON file and a Markdown table under
`correlations/webnlg17/results/`; use `--output-dir` to write them elsewhere.

### Appendix Table 18 scores

All values below are percentages, rounded to one decimal place.

| Model | System r | System rho | System tau | Text r | Text rho | Text tau |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BLEU | 77.0 | 71.6 | 56.8 | 77.6 | 79.6 | 66.9 |
| METEOR | 86.8 | 83.5 | 67.8 | 79.9 | 80.2 | 69.3 |
| PARENT | 88.2 | 82.6 | 66.7 | 79.2 | 79.6 | 66.9 |
| BERTScore | 70.1 | 78.3 | 63.7 | 80.0 | 81.1 | 69.5 |
| BARTScore | 91.0 | 87.5 | 78.2 | 81.4 | 81.8 | 70.9 |
| BLEURT | 90.5 | 88.1 | 72.1 | 82.1 | 82.6 | 71.4 |
| Data-QuestEval-F1 | 94.9 | 93.4 | 85.9 | 78.1 | 79.3 | 67.6 |
| FactSpotter | 97.2 | 93.9 | 85.1 | 84.7 | 81.8 | 70.3 |
| NLI-F1 | 94.5 | 95.0 | 87.0 | 82.6 | 78.5 | 68.3 |
| MonoLR-F1 | 95.1 | 89.7 | 77.9 | 84.1 | 81.4 | 70.4 |
| Gemma3-270M-F1 | 94.7 | 88.5 | 79.4 | 83.9 | 82.3 | 75.1 |
| Gemma3-1B-F1 | 93.0 | 86.5 | 76.4 | 83.5 | 82.2 | 74.8 |
| Gemma3-4B-F1 | 93.2 | 87.7 | 78.1 | 82.5 | 82.1 | 74.7 |
| Gemma3-12B-F1 | 91.6 | 87.7 | 78.0 | 83.5 | 82.6 | 75.2 |
| Qwen3-0.6B-F1 | 92.9 | 87.8 | 78.2 | 83.2 | 82.9 | 75.8 |
| Qwen3-1.7B-F1 | 92.4 | 86.9 | 77.1 | 83.6 | 83.1 | 75.5 |
| Qwen3-4B-F1 | 91.3 | 85.2 | 74.2 | 82.2 | 82.6 | 75.3 |
| Qwen3-8B-F1 | 91.1 | 85.2 | 73.4 | 82.4 | 82.5 | 75.3 |
| Qwen3-14B-F1 | -- | -- | 58.3 | 80.9 | 76.7 | 69.2 |
| Llama3.2-1B-F1 | 91.8 | 86.9 | 76.3 | 83.6 | 82.4 | 75.3 |
| Llama3.2-3B-F1 | 92.0 | 86.9 | 76.8 | 82.8 | 83.0 | 75.7 |
| Llama3.1-8B-F1 | 92.2 | 86.5 | 76.1 | 82.2 | 81.7 | 74.9 |

## Evaluation Protocol

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

`metric_results/` holds the example-level scores needed to reproduce the
paper table: BLEU, METEOR, PARENT, BERTScore, BARTScore, BLEURT,
Data-QuestEval, FactSpotter, NLI, MonoLR, and the 12 XQDT checkpoints.
`score_webnlg17.py` maps each paper label to its directory through the
`WEBLG17_METRICS` constant.

## Sanity Checks

From the repository root:

```bash
ls correlations/webnlg17/webnlg2017_inputs/teams-generations
ls correlations/webnlg17/human-annotations/semantics
ls correlations/webnlg17/metric_results
```
