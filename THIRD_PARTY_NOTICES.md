# Third-party notices

The components below retain their original licenses and attribution. The root GPLv3 license applies to XQDT code, not to third-party datasets or model weights.

## PARENT

- Paths: `repair_d2t/metrics/parent/` and `correlations/e2e19/baselines/parent_utils.py`
- Upstream: [KaijuML/parent](https://github.com/KaijuML/parent) and [Google Research's table-to-text evaluation code](https://github.com/google-research/language/tree/master/language/table_text_eval)
- License: Apache License 2.0
- License copy: [Apache-2.0](repair_d2t/metrics/parent/THIRD_PARTY_LICENSE_APACHE_2.0.txt)

The source credits the Google AI Language Team Authors (2018) and Clément Rebuffel (2021). The E2E helper extracts PARENT functions from the Google Research implementation.

## chrF / chrF++

- Path: `repair_d2t/metrics/chrF.py`
- Upstream: [Maja Popović's chrF](https://github.com/m-popovic/chrF)
- License: GNU GPL; the source header does not specify a version.
- License notice and citation: preserved in the [source header](repair_d2t/metrics/chrF.py).

The implementation credits Maja Popović (2017) and requests citation of *chrF: character n-gram F-score for automatic MT evaluation* (WMT 2015).

## WebNLG 2020 Text-to-RDF evaluator

- Path: `repair_t2d/eval/eval_webnlg_t2g/`
- Upstream task: <https://webnlg-challenge.loria.fr/challenge_2020/>
- License: GNU General Public License v3.0
- License copy: [GPLv3](repair_t2d/eval/eval_webnlg_t2g/LICENSE)

## QuestEval

- Paths: `questeval_metric.py` and `quest_utils.py` under `correlations/4L-RP-Human/baselines/`, `correlations/e2e19/baselines/`, and `correlations/webnlg20composed/baselines/`
- Upstream: <https://github.com/ThomasScialom/QuestEval>
- License: MIT License
- License copy: [MIT](third_party_licenses/QuestEval-LICENSE.txt)

These files adapt the upstream QuestEval implementation to the evaluation workflows in this repository.

## 4L-RP-Human annotations

- Paths: `correlations/4L-RP-Human/source_annotations/` and `correlations/4L-RP-Human/by_language/`
- Reference: Soto Martinez et al., [Semantic Evaluation of Multilingual Data-to-Text Generation via NLI Fine-Tuning](https://aclanthology.org/2025.findings-acl.542/)
- Source project: <https://gitlab.inria.fr/wsotomar/semantic-evaluation-of-multilingual-data-to-text-generation-via-nli-fine-tuning>

The reported 4L-RP-Human benchmark covers English, Maltese, Russian, and Welsh. The release here also contains the available Arabic, Chinese, French, and Spanish annotation subsets.

## WebNLG 2017 human evaluation

- Paths: `correlations/webnlg17/human-annotations/` and `correlations/webnlg17/webnlg2017_inputs/`
- Upstream: <https://gitlab.com/webnlg/webnlg-human-evaluation>
- Reference: Gardent et al., *The WebNLG Challenge: Generating Text from RDF Data* (2017)

The release contains aggregate semantic-adequacy scores, source triples, and system outputs from the original human-evaluation sample.
