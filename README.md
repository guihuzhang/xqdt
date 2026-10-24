# XQDT

Code for **XQDT: eXplainable and Quantitative Data-Text Alignment Metric with Feedback Signals** (EMNLP 2026 Main Conference).

XQDT checks the alignment between structured data and text. It identifies omitted, extra, and incorrect information and combines these decisions into precision, recall, and F1 scores. The detected errors can also guide the correction of generated text or extracted data.

The repository includes training and evaluation data, model predictions, repair outputs, and code for the experiments below.

## Models and Data

The 24 XQDT LoRA adapters are available on Hugging Face:

- [WebNLG models](https://huggingface.co/collections/Loria-MosAIk/xqdt-webnlg-models-6a94697a0582edd20ad3df04): 12 verifiers trained on WebNLG synthetic data.
- [E2E models](https://huggingface.co/collections/Loria-MosAIk/xqdt-e2e-models-6a94697d5be8186d7795da71): 12 verifiers trained on the combined WebNLG and E2E synthetic data.

The synthetic training, development, and test data are included in the [WebNLG](verifier_train_eval_webnlg/data/README.md) and [E2E](verifier_train_eval_e2e19/data/README.md) directories. The processed [4L-RP-Human annotations](https://huggingface.co/datasets/Loria-MosAIk/4L-RP-Human-Clean) are also available on Hugging Face.

## Experiments

Each directory contains its results, inputs, saved predictions, and instructions for scoring or running models.

| Paper result | Directory |
| --- | --- |
| Table 3: WebNLG verifier results | [WebNLG verifier](verifier_train_eval_webnlg/README.md) |
| Table 4: E2E verifier results | [E2E verifier](verifier_train_eval_e2e19/README.md) |
| Table 5: 4L-RP-Human correlations | [4L-RP-Human](correlations/4L-RP-Human/README.md) |
| Table 6: WebNLG 2020 correlations | [WebNLG 2020](correlations/webnlg20composed/README.md) |
| Table 7: E2E coarse-label evaluation | [E2E human evaluation](correlations/e2e19/README.md) |
| Table 8 and Appendix Tables 21--24: KELM | [KELM](correlations/kelm/README.md) |
| Tables 9--10: text-to-data repair | [T2D repair](repair_t2d/README.md) |
| Table 11: data-to-text repair | [D2T repair](repair_d2t/README.md) |
| Appendix: WebNLG 2017 correlations | [WebNLG 2017](correlations/webnlg17/README.md) |

For the repair experiments, the shared inputs and per-round records reconstruct the complete outputs used for scoring. The T2D and D2T instructions cover both the main-text and appendix tables.

## Getting Started

Use Python 3.11 and run the commands from the repository root. To recompute the verifier scores from the saved predictions:

```bash
python -m pip install -r correlations/requirements.txt

python verifier_train_eval_webnlg/score_verifier_release.py \
  --dataset webnlg --output-dir verifier_train_eval_webnlg/evaluation_results

python verifier_train_eval_webnlg/score_verifier_release.py \
  --dataset e2e19 --output-dir verifier_train_eval_e2e19/evaluation_results
```

These commands score all 12 fine-tuned verifiers and five prompted baselines on each dataset using strict and element matching. Each scorer writes detailed metrics and a Markdown result table. Add `--similarity` after installing the embedding dependencies described in the verifier README to include similarity matching.

For inference and training, follow the [WebNLG verifier](verifier_train_eval_webnlg/README.md#run-xqdt-inference) or [E2E verifier](verifier_train_eval_e2e19/README.md#run-xqdt-inference) instructions. The model cards also provide examples for using individual adapters. Correlation and repair dependencies are listed in their respective directories.

## Citation

```bibtex
@inproceedings{efimov-zhang-etal-2026-xqdt,
  title = {{XQDT}: eXplainable and Quantitative Data-Text Alignment Metric with Feedback Signals},
  author = {Efimov-Zhang, Kun and Song, Yifei and Gardent, Claire},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year = {2026},
  publisher = {Association for Computational Linguistics}
}

@inproceedings{zhang-etal-2023-factspotter,
  title = {{F}act{S}potter: Evaluating the Factual Faithfulness of Graph-to-Text Generation},
  author = {Zhang, Kun and Balalau, Oana and Manolescu, Ioana},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2023},
  year = {2023},
  month = dec,
  address = {Singapore},
  publisher = {Association for Computational Linguistics},
  pages = {10025--10042},
  doi = {10.18653/v1/2023.findings-emnlp.672},
  url = {https://aclanthology.org/2023.findings-emnlp.672/}
}
```

## License

XQDT code is licensed under [GNU GPLv3](LICENSE) (`GPL-3.0-only`).

Third-party code retains its original licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
