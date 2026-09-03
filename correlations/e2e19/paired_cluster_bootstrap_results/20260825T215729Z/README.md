# Paired Bootstrap: WebNLG-only to E2E Transfer

This confirmatory analysis compares WebNLG-only training with WebNLG+E2E training on identical E2E examples. Delta is `WebNLG-only - WebNLG+E2E`. Coarse-label F1 and text-level correlations use paired MR-cluster resampling. System-level correlations are handled separately with paired system resampling.

- Bootstrap replicates: 10,000
- Seed: 42
- Confidence interval: percentile 95%
- Git commit: `739c84d868f9d32bcf5c150bdf05c56260cfcaf4` (dirty: `True`)

## Key Results

| Model | Level | Metric | Joint | Cross | Delta | 95% CI | CI excludes 0 |
|---|---|---|---:|---:|---:|---:|:---:|
| Qwen3-4B | coarse_label | is_ok_f1 | 89.36 | 89.83 | +0.46 | [+0.20, +0.73] | yes |
| Qwen3-4B | coarse_label | has_missing_f1 | 75.82 | 76.42 | +0.60 | [+0.08, +1.12] | yes |
| Qwen3-4B | coarse_label | has_added_f1 | 21.09 | 23.94 | +2.86 | [-4.20, +9.55] | no |
| Qwen3-4B | text_correlation | f1_spearman | 51.52 | 51.39 | -0.13 | [-1.08, +0.76] | no |
| Qwen3-8B | coarse_label | is_ok_f1 | 89.39 | 89.22 | -0.17 | [-0.63, +0.21] | no |
| Qwen3-8B | coarse_label | has_missing_f1 | 75.74 | 75.22 | -0.52 | [-1.46, +0.28] | no |
| Qwen3-8B | coarse_label | has_added_f1 | 23.33 | 20.71 | -2.61 | [-9.63, +4.26] | no |
| Qwen3-8B | text_correlation | f1_spearman | 51.38 | 51.21 | -0.18 | [-0.98, +0.60] | no |

## Interpretation Rule

A 95% CI containing zero is reported as comparable / not reliably distinguishable under this resampling design. A CI excluding zero supports a directional difference for that metric only.

## Files

- `results.json`: complete machine-readable results
- `results.csv`: flat result table
- `table.tex`: paste-ready LaTeX table
The private execution manifest retains machine-local paths and is not included in the public Git repository. Input hashes and the full manifest will accompany the frozen artifact bundle.
