# WebNLG+E2E verifier data

This directory contains synthetic data-text pairs and error labels for training
and evaluating the E2E verifiers. The training set (44,611 examples) and
validation set (2,785 examples) combine WebNLG and E2E data. The test set
contains 1,229 E2E examples.

The shared [synthetic-data audit](../../verifier_train_eval_webnlg/data/synthetic_audit/README.md)
contains 180 manually checked examples: 90 from WebNLG and 90 from E2E. It
includes the final human decisions and a script to reproduce the audit results.
