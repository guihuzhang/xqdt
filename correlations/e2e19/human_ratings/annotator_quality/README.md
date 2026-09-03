# Annotator quality diagnostics

This directory preserves the annotator-quality analysis used in the paper.
Annotator identifiers are stable within this release.

The diagnostic groups are:

- **A**: repeated-key input error
- **B**: out-of-range score; individual scores above 100 are removed
- **C**: coarse/fine label mismatch
- **D**: fixed-value rating pattern
- **E**: ceiling-effect rating pattern
- **F**: wrong rating scale

The publication-time filter excludes annotators in A, C, D, E, and F. Group B
is handled at annotation level by removing scores above 100.

`annotator_quality_flags.csv` provides annotator-level flags and filtering
status. `flagged_annotations.csv` provides the associated rating cases.
