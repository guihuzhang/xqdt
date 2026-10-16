# Synthetic Data Audit

We manually checked 180 synthetic data-text pairs: 90 from WebNLG and 90 from
E2E, with 60 examples for each error type. For each pair, we assessed whether
the intended perturbation and its error label were valid (validity), and
whether all alignment errors were labelled (completeness).

`synthetic_audit_adjudicated.csv` contains the pairs, their generated error
labels, and the final human decisions.

| Error type | Perturbation validity | Label completeness |
| --- | ---: | ---: |
| Extra | 55/60 | 49/60 |
| Omitted | 57/60 | 52/60 |
| Incorrect | 57/60 | 53/60 |
| Overall | 169/180 (93.9%) | 154/180 (85.6%) |

Run from the repository root:

```bash
python verifier_train_eval_webnlg/data/synthetic_audit/summarize_audit.py
```
