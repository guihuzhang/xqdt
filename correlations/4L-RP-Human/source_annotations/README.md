# Source Annotations

These CSV files are a de-identified source view of the annotation export used to prepare the language-level evaluation inputs. The source ratings remain unchanged in `annotations.csv`; subsequent selection and correction steps are recorded separately.

- `items.csv` contains the text, graph, language, level, and public item ID.
- `annotations.csv` contains the original precision and recall ratings linked by public item and annotator IDs.
- `annotators.csv` records which annotators were retained for the evaluation inputs and their order within each language.
- `annotation_corrections.csv` records the historical score correction needed to reproduce the processed English file.

## From Source Export to Evaluation Files

| Stage | Items | Annotators | Annotations | Contents |
| --- | ---: | ---: | ---: | --- |
| De-identified source export | 572 | 64 | 2,371 | All collected levels and annotators |
| Language-level evaluation files | 400 | 24 | 1,200 | 50 level-2 items per language and the retained evaluation annotators |

`../prepare_annotations.py` performs four steps:

1. selects the level-2 items;
2. retains the annotators marked in `annotators.csv`, preserving their recorded order;
3. applies the correction listed in `annotation_corrections.csv`;
4. maps ratings to the evaluation scale with `(rating - 1) / 4` and writes the eight files in `../by_language/`.

## Recorded Correction

Exactly one historical value correction is applied:

| Item | Annotator | Field | Source value | Corrected value | Evaluation value |
| --- | --- | --- | ---: | ---: | ---: |
| `English_173` | `annotator_011` | precision | 0 | 5 | 1.0 |

The correction is stored in machine-readable form in `annotation_corrections.csv`. The source value remains unchanged in `annotations.csv`.

## Preserved Out-of-Range Values

Within the selected level-2 evaluation annotations, three other rating fields have value `0`, outside the documented 1--5 scale. No historical correction was recorded for them, so they are preserved rather than guessed:

| Item | Evaluation item | Annotator | Field | Source value | Evaluation value |
| --- | --- | --- | --- | ---: | ---: |
| `Arabic_470` | `Arabic_0` | `annotator_019` | precision | 0 | -0.25 |
| `Russian_229` | `Russian_9` | `annotator_031` | precision | 0 | -0.25 |
| `Russian_229` | `Russian_9` | `annotator_031` | recall | 0 | -0.25 |

The source tables also contain a level-1 value of `0` for the recall rating of `English_64` by `annotator_008`. It is excluded with the other non-level-2 annotations and therefore does not appear in any evaluation file. No other rating values are changed. This makes the historical evaluation inputs reproducible without introducing undocumented repairs.

The IDs in this directory are repository-local. Prolific identifiers, database row IDs, and response times are not included.
