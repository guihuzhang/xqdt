#!/usr/bin/env python3
"""Re-run the WebNLG20 legacy baselines and compare against bundled scores."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
REPO_ROOT = SCRIPT_DIR.parents[2]

DEFAULT_TEAM_DIR = PACKAGE_DIR / "webnlg2020_inputs" / "team-generations"
DEFAULT_TRIPLES_FILE = PACKAGE_DIR / "webnlg2020_inputs" / "triples_modified.txt"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reproduced_metric_results"
DEFAULT_GOLDEN_DIR = PACKAGE_DIR / "metric_results"
DEFAULT_TEMPLATE_FILE = SCRIPT_DIR / "webnlg_templates.json"
DEFAULT_FACTSPOTTER_CHECKPOINT = REPO_ROOT / "checkpoints" / "fact_spotter_electra.pt"

BASELINE_FOLDERS = {
    "nli": ("nli_based_p", "nli_based_r", "nli_based_f1"),
    "monolr": ("monolr-precision", "monolr-recall", "monolr-f1"),
    "factspotter": ("fact-spotter",),
    "dqe": ("questeval_p", "questeval_r", "questeval"),
}

TEMPLATE_REMAP = {
    "eat_type": "eatType",
    "rating": "customer rating",
    "family_friendly": "familyFriendly",
    "price_range": "priceRange",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(BASELINE_FOLDERS),
        default=list(BASELINE_FOLDERS),
        help="Baselines to run (default: all).",
    )
    parser.add_argument("--team-dir", type=Path, default=DEFAULT_TEAM_DIR)
    parser.add_argument("--triples-file", type=Path, default=DEFAULT_TRIPLES_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--golden-dir", type=Path, default=DEFAULT_GOLDEN_DIR)
    parser.add_argument("--template-file", type=Path, default=DEFAULT_TEMPLATE_FILE)
    parser.add_argument(
        "--factspotter-checkpoint",
        type=Path,
        default=DEFAULT_FACTSPOTTER_CHECKPOINT,
    )
    parser.add_argument("--limit", type=int, default=None, help="Use only the first N entries per system.")
    parser.add_argument("--dqe-batch-size", type=int, default=16)
    parser.add_argument("--no-compare", action="store_true")
    return parser.parse_args()


def require_path(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def parse_triples(line: str) -> List[str]:
    return [part.strip() for part in line.split("<br>") if part.strip()]


def load_inputs(team_dir: Path, triples_file: Path, limit: int | None) -> Tuple[List[List[str]], Dict[str, List[str]]]:
    triples_file = require_path(triples_file, "triples file")
    team_dir = require_path(team_dir, "team directory")

    triple_lines = triples_file.read_text(encoding="utf-8").splitlines()
    triples = [parse_triples(line) for line in triple_lines]
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive")
        triples = triples[:limit]

    teams: Dict[str, List[str]] = {}
    for path in sorted(team_dir.glob("*.txt")):
        texts = path.read_text(encoding="utf-8").splitlines()
        if limit is not None:
            texts = texts[:limit]
        if len(texts) != len(triples):
            raise ValueError(f"{path.name}: {len(texts)} texts for {len(triples)} triple sets")
        teams[path.stem] = texts

    if not teams:
        raise ValueError(f"No team .txt files found under {team_dir}")
    return triples, teams


def write_scores(output_dir: Path, folder: str, scores_by_team: Dict[str, Sequence[float]]) -> None:
    target = output_dir / folder
    target.mkdir(parents=True, exist_ok=True)
    for team, scores in scores_by_team.items():
        (target / f"{team}.txt").write_text(
            "\n".join(repr(float(score)) for score in scores) + "\n",
            encoding="utf-8",
        )


def harmonic(p: float, r: float) -> float:
    return 0.0 if p + r == 0 else 2.0 * p * r / (p + r)


# ---------------------------------------------------------------------------
# NLI baseline (Dusek & Kasner): MR->text precision, text->triples recall.
# ---------------------------------------------------------------------------


def load_templates(path: Path) -> dict:
    with open(require_path(path, "NLI template file"), "r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_triple_dict(triple: str) -> dict | None:
    parts = [part.strip() for part in triple.split("|")]
    if len(parts) != 3:
        return None
    return {"subject": parts[0], "predicate": parts[1], "object": parts[2]}


def triple_to_template(triple: dict, templates: dict) -> str:
    predicate = triple["predicate"]
    subject = triple["subject"]
    obj = triple["object"]

    if predicate in templates:
        template = templates[predicate]
    elif predicate in TEMPLATE_REMAP and TEMPLATE_REMAP[predicate] in templates:
        template = templates[TEMPLATE_REMAP[predicate]]
    else:
        template = f"The {predicate} of <subject> is <object>."
        template = re.sub(r"([a-z])([A-Z])", r"\1 \2", template)

    if isinstance(template, dict):
        template = template.get(obj, next(iter(template.values())))
    if isinstance(template, list):
        template = template[0]

    template = template.replace("<subject>", subject)
    obj = re.sub(r"^[\"'](.*)[\"']$", r"\1", obj)
    return template.replace("<object>", obj).replace("_", " ")


def run_nli(triples: List[List[str]], teams: Dict[str, List[str]], template_file: Path) -> Dict[str, Dict[str, List[float]]]:
    import numpy as np
    import torch
    from transformers import RobertaForSequenceClassification, RobertaTokenizer

    tokenizer = RobertaTokenizer.from_pretrained("roberta-large-mnli")
    model = RobertaForSequenceClassification.from_pretrained("roberta-large-mnli")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    templates = load_templates(template_file)

    def entailment(premise: str, hypothesis: str) -> float:
        encoded = tokenizer(f"{premise} </s></s> {hypothesis}", return_tensors="pt", truncation=True)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits[0]
        return float(torch.softmax(logits, dim=-1)[2].item())

    out = {folder: {} for folder in BASELINE_FOLDERS["nli"]}
    for team, texts in teams.items():
        p_scores: List[float] = []
        r_scores: List[float] = []
        f_scores: List[float] = []
        for text, graph in zip(texts, triples):
            templated = [triple_to_template(parsed, templates) for raw in graph if (parsed := parse_triple_dict(raw))]
            if not templated:
                p = r = 0.0
            else:
                p = entailment(" ".join(templated), text)
                r = float(np.mean([entailment(text, sentence) for sentence in templated]))
            p_scores.append(p)
            r_scores.append(r)
            f_scores.append(harmonic(p, r))
        out["nli_based_p"][team] = p_scores
        out["nli_based_r"][team] = r_scores
        out["nli_based_f1"][team] = f_scores
    return out


# ---------------------------------------------------------------------------
# MonoLR baseline. Keep the historical WebNLG20 formatting and direction.
# ---------------------------------------------------------------------------


def clean_entity(entity: str) -> str:
    entity = re.sub(r"@[a-z]{2,3}$", "", entity)
    entity = re.sub(r"\^\^xsd:\w+", "", entity)
    entity = re.sub(r"\^\^<http://[^>]+>", "", entity)
    return entity.strip("\"'").replace("_", " ").strip()


def clean_predicate(predicate: str) -> str:
    predicate = predicate.replace("/", " ")
    predicate = re.sub(r"([a-z])([A-Z])", r"\1 \2", predicate)
    predicate = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", predicate)
    return predicate.strip().lower()


def format_monolr_graph(graph: Iterable[str]) -> str:
    formatted: List[str] = []
    for raw in graph:
        parts = [part.strip() for part in raw.split("|")]
        if len(parts) != 3:
            continue
        subject, predicate, obj = parts
        formatted.append(
            f"[S]{clean_entity(subject)}[P]{clean_predicate(predicate)}[O]{clean_entity(obj)}"
        )
    return "[T]".join(formatted)


def run_monolr(triples: List[List[str]], teams: Dict[str, List[str]]) -> Dict[str, Dict[str, List[float]]]:
    import torch
    from sentence_transformers import CrossEncoder

    model = CrossEncoder("MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")
    model.config.num_labels = 1
    model.model.load_adapter("WilliamSotoM/MonoLR_eng_Latn_PR")

    out = {folder: {} for folder in BASELINE_FOLDERS["monolr"]}
    for team, texts in teams.items():
        p_scores: List[float] = []
        r_scores: List[float] = []
        f_scores: List[float] = []
        for text, graph in zip(texts, triples):
            graph_text = format_monolr_graph(graph)
            if not graph_text or not text:
                p = r = 0.0
            else:
                p_logit = model.predict([(graph_text, text)])[0]
                r_logit = model.predict([(text, graph_text)])[0]
                p = float(torch.sigmoid(torch.as_tensor(p_logit)).item())
                r = float(torch.sigmoid(torch.as_tensor(r_logit)).item())
            p_scores.append(p)
            r_scores.append(r)
            f_scores.append(harmonic(p, r))
        out["monolr-precision"][team] = p_scores
        out["monolr-recall"][team] = r_scores
        out["monolr-f1"][team] = f_scores
    return out


# ---------------------------------------------------------------------------
# Original FactSpotter: ELECTRA-small binary classifier, recall only.
# ---------------------------------------------------------------------------


def run_factspotter(
    triples: List[List[str]],
    teams: Dict[str, List[str]],
    checkpoint: Path,
) -> Dict[str, Dict[str, List[float]]]:
    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from unidecode import unidecode

    checkpoint = require_path(checkpoint, "FactSpotter ELECTRA checkpoint")
    base_model = "google/electra-small-discriminator"
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=2)
    try:
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint, map_location="cpu")
    # strict=False: the checkpoint carries the non-persistent position_ids buffer,
    # which recent transformers versions no longer include in a fresh state_dict.
    model.load_state_dict(state_dict, strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    def triple_score(text: str, raw: str) -> float:
        parts = [part.strip() for part in raw.split("|")]
        if len(parts) != 3:
            return 0.0
        subject, predicate, obj = [unidecode(part.lower().strip()) for part in parts]
        normalized_text = unidecode(text.lower().strip())
        formatted = f"predicate: {predicate}, subject: {subject}, object: {obj}, sentence: {normalized_text}"
        encoded = tokenizer([formatted], truncation=True, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits[0]
        return float(torch.softmax(logits, dim=-1)[1].item())

    out = {"fact-spotter": {}}
    for team, texts in teams.items():
        scores: List[float] = []
        for text, graph in zip(texts, triples):
            per_triple = [triple_score(text, raw) for raw in graph]
            scores.append(float(np.mean(per_triple)) if per_triple else 0.0)
        out["fact-spotter"][team] = scores
    return out


# ---------------------------------------------------------------------------
# Data-QuestEval: preserve the historical private-score extraction.
# ---------------------------------------------------------------------------


def run_dqe(
    triples: List[List[str]],
    teams: Dict[str, List[str]],
    output_dir: Path,
    batch_size: int,
) -> Dict[str, Dict[str, List[float]]]:
    import torch

    from questeval_metric import QuestEval

    questeval = QuestEval(
        task="data2text",
        language="en",
        use_cache=True,
        no_cuda=not torch.cuda.is_available(),
    )
    log_dir = output_dir / ".questeval_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    questeval.log_dir = str(log_dir)
    questeval.hash_files = set(path.name for path in log_dir.iterdir())

    out = {folder: {} for folder in BASELINE_FOLDERS["dqe"]}
    for team, texts in teams.items():
        questeval.corpus_questeval(
            hypothesis=texts,
            sources=triples,
            list_references=None,
            batch_size=batch_size,
        )
        p_scores: List[float] = []
        r_scores: List[float] = []
        f_scores: List[float] = []
        for text, graph in zip(texts, triples):
            src_text = questeval.src_preproc_pipe(graph) if questeval.src_preproc_pipe else " ".join(graph)
            hyp_log = questeval.open_log_from_text(text)
            src_log = questeval.open_log_from_text(src_text)
            p = float(questeval._base_score(src_log, hyp_log))
            r = float(questeval._base_score(hyp_log, src_log))
            p_scores.append(p)
            r_scores.append(r)
            f_scores.append(harmonic(p, r))
        out["questeval_p"][team] = p_scores
        out["questeval_r"][team] = r_scores
        out["questeval"][team] = f_scores
    return out


def merge_outputs(destination: Dict[str, Dict[str, List[float]]], source: Dict[str, Dict[str, List[float]]]) -> None:
    for folder, scores in source.items():
        destination[folder] = scores


def read_float_lines(path: Path) -> List[float]:
    return [float(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def compare_with_golden(
    output_dir: Path,
    golden_dir: Path,
    folders: Sequence[str],
    limit: int | None,
) -> None:
    print("\nGolden-cache comparison")
    print("folder\tcount\tmax_abs_diff\tmean_abs_diff")
    for folder in folders:
        diffs: List[float] = []
        generated_dir = output_dir / folder
        golden_folder = golden_dir / folder
        if not golden_folder.exists():
            print(f"{folder}\tMISSING_GOLDEN")
            continue
        for generated_path in sorted(generated_dir.glob("*.txt")):
            golden_path = golden_folder / generated_path.name
            if not golden_path.exists():
                raise FileNotFoundError(f"Golden file missing: {golden_path}")
            generated = read_float_lines(generated_path)
            golden = read_float_lines(golden_path)
            if limit is not None:
                golden = golden[:limit]
            if len(generated) != len(golden):
                raise ValueError(
                    f"{folder}/{generated_path.name}: generated {len(generated)} scores, golden {len(golden)}"
                )
            diffs.extend(abs(a - b) for a, b in zip(generated, golden))
        max_diff = max(diffs) if diffs else math.nan
        mean_diff = sum(diffs) / len(diffs) if diffs else math.nan
        print(f"{folder}\t{len(diffs)}\t{max_diff:.12g}\t{mean_diff:.12g}")


def main() -> None:
    args = parse_args()
    triples, teams = load_inputs(args.team_dir, args.triples_file, args.limit)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_outputs: Dict[str, Dict[str, List[float]]] = {}
    for baseline in args.models:
        print(f"Running {baseline} on {len(teams)} systems x {len(triples)} entries")
        if baseline == "nli":
            result = run_nli(triples, teams, args.template_file)
        elif baseline == "monolr":
            result = run_monolr(triples, teams)
        elif baseline == "factspotter":
            result = run_factspotter(triples, teams, args.factspotter_checkpoint)
        elif baseline == "dqe":
            result = run_dqe(triples, teams, args.output_dir, args.dqe_batch_size)
        else:
            raise AssertionError(baseline)
        merge_outputs(all_outputs, result)
        for folder, scores_by_team in result.items():
            write_scores(args.output_dir, folder, scores_by_team)

    if not args.no_compare:
        compare_with_golden(
            args.output_dir,
            require_path(args.golden_dir, "golden metric directory"),
            list(all_outputs),
            args.limit,
        )


if __name__ == "__main__":
    main()
