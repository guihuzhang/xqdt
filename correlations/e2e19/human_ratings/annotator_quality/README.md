# E2E annotation issues and cleaning

## Original evaluation

The human ratings come from the E2E NLG Challenge evaluation described in
Dušek, Novikova and Rieser's
[Evaluating the State-of-the-Art of End-to-End Natural Language Generation: The E2E NLG Challenge](https://arxiv.org/abs/1901.07931)
(2020). The [original repository](https://github.com/tuetschek/e2e-eval)
provides the participating systems' restaurant descriptions, their input
meaning representations (MRs), and the crowdsourced ratings. Our local
`../converted.json` groups these records by MR and system, covering 6,012
input–output pairs.

## What we found

Before using these annotations to evaluate XQDT, we inspected the numeric
quality scores and coarse error labels. Quality scores use a 0–100 scale;
`missing` marks input information that is not correctly expressed, `added`
marks unsupported information in the text, and `ok` marks no semantic error.
Our analysis identified the following issues and rating patterns:

| Group | Observation in the ratings | How we handle it |
| --- | --- | --- |
| A | Repeated-digit entries such as 111 or 111111 | Exclude the flagged annotator's ratings |
| B | Individual scores above 100, such as 105 or 110 | Remove those scores; keep the annotator's other ratings |
| C | Numeric scores and coarse labels disagree, for example 100 together with `missing` or `added` | Exclude the flagged annotator's ratings |
| D | The same score or a very limited set of scores across different outputs, for example repeatedly giving 21 | Exclude the flagged annotator's ratings |
| E | Always or almost always assigning 100 across different outputs | Exclude the flagged annotator's ratings |
| F | Scores concentrated on 1–4 rather than the intended 0–100 scale | Exclude the flagged annotator's ratings |

## What we changed

We apply these filters when scoring, without rewriting the source ratings in
`../converted.json`:

1. Exclude all ratings from the 23 annotators listed in
   `../excluded_workers.txt`. This list contains 21 annotators from groups
   A, C, D, E, and F, plus two recorded as `publication_exclusion` without
   a group in the exported analysis.
2. Remove any remaining individual score above 100. Group B annotators are
   retained with their other ratings.
3. For correlations, average the remaining quality scores for each MR–system
   pair. Omit pairs with no remaining ratings.

This leaves 5,515 of the original 6,012 input–output pairs for evaluation.

## Files

- `annotator_quality_flags.csv`: the groups and exclusion decisions for 26
  annotators.
- `flagged_annotations.csv`: 3,524 rating records collected for inspecting
  flagged annotators, with the input, system output, numeric score, coarse
  label, and notes.
- `../converted.json`: the source ratings grouped by MR and system.
- `../excluded_workers.txt`: the exclusion list read by the scoring scripts.
- `../../worker_filter.py`: reads the exclusion list.
- `../../e2e_correlation_light.py`: applies the filters and computes correlations.
