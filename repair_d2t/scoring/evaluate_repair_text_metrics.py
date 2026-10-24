#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate original/repaired D2T texts with lightweight text metrics."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from sacrebleu.metrics import BLEU, CHRF, TER
from rouge_score import rouge_scorer

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_METRICS_DIR = PROJECT_DIR / "metrics"
DEFAULT_METEOR_JAR = DEFAULT_METRICS_DIR / "meteor-1.5/meteor-1.5.jar"
DEFAULT_METRICS = "bertscore,bleu,chrf++,meteor,parent,rouge_l,sbert"
METRIC_OUTPUT_KEYS = {
    "bertscore": {"bertscore_precision", "bertscore_recall", "bertscore_f1"},
    "bleu": {"bleu"},
    "chrf++": {"chrf++"},
    "meteor": {"meteor"},
    "parent": {"parent_precision", "parent_recall", "parent_f1"},
    "rouge_l": {"rouge_l"},
    "sbert": {"sbert_mpnet"},
}
DEFAULT_NLI_MODEL = "roberta-large-mnli"
DEFAULT_NLI_TEMPLATE_FILE = DEFAULT_METRICS_DIR / "nli_dusek" / "webnlg_templates.json"
TEMPLATE_REMAP = {
    "eat_type": "eatType",
    "rating": "customer rating",
    "family_friendly": "familyFriendly",
    "price_range": "priceRange",
}
METEOR_LANGS = {"en", "ru"}

try:
    import pyter
except Exception:  # pragma: no cover - optional fallback
    pyter = None

try:
    import nltk
except Exception:  # pragma: no cover - optional fallback
    nltk = None

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional fallback
    tqdm = None


def log_start(label: str) -> float:
    print(f"{label} start", flush=True)
    return time.perf_counter()


def log_end(label: str, start: float) -> None:
    print(f"{label} end in {time.perf_counter() - start:.2f}s", flush=True)


def progress_iter(iterable, *, desc: str, unit: str = "sample", leave: bool = False):
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, unit=unit, leave=leave)


def to_percentage(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    if abs(value) <= 1.000001:
        return float(value * 100.0)
    return float(value)


def to_percentage_dict(values: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
    if values is None:
        return None
    return {key: to_percentage(value) for key, value in values.items()}


def ensure_file(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")
    return path


def collect_input_files(paths: Iterable[Path]) -> List[Path]:
    files: List[Path] = []
    for path in paths:
        path = path.expanduser().resolve()
        if path.is_dir():
            files.extend(sorted(path.glob("*.json")))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(f"Input path not found: {path}")
    return files


def extract_rows(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        return payload["results"]
    raise ValueError("Expected JSON list or dict with a 'results' list")


def references_for_row(row: Dict[str, Any]) -> List[str]:
    refs = row.get("reference_texts") or []
    if refs:
        return [str(ref).replace("\n", " ").strip() for ref in refs if str(ref).strip()]
    ref = str(row.get("reference_text", "")).replace("\n", " ").strip()
    return [ref] if ref else []


def collect_texts(rows: List[Dict[str, Any]], text_key: str) -> Dict[str, List[str]]:
    refs: List[List[str]] = []
    hyps: List[str] = []
    for row in rows:
        row_refs = references_for_row(row)
        hyp = str(row.get(text_key, "")).replace("\n", " ").strip()
        if not hyp and text_key == "input_text":
            hyp = str(row.get("original_text", "")).replace("\n", " ").strip()
        if not hyp and text_key == "repaired_text":
            hyp = str(row.get("original_text") or row.get("input_text", "")).replace("\n", " ").strip()
        if row_refs:
            refs.append(row_refs)
            hyps.append(hyp)
    return {"refs": refs, "hyps": hyps}


def collect_graphs(rows: List[Dict[str, Any]]) -> List[List[str]]:
    graphs: List[List[str]] = []
    for row in rows:
        triples = row.get("source_triples") or row.get("raw_source_triples") or []
        graphs.append([str(triple) for triple in triples])
    return graphs


def trim_graphs_for_refs(graphs: List[List[str]], refs_count: int) -> List[List[str]]:
    return graphs[:refs_count]


def parse_metrics(metrics: str, include_meteor: bool) -> Set[str]:
    aliases = {
        "bert": "bertscore",
        "bert_score": "bertscore",
        "mpnet": "sbert",
        "sentencebert": "sbert",
        "sentence_bert": "sbert",
        "rouge": "rouge_l",
        "rougel": "rouge_l",
        "rouge-l": "rouge_l",
    }
    names = {item.strip().lower() for item in metrics.split(",") if item.strip()}
    names = {aliases.get(name, name) for name in names}
    if include_meteor:
        names.add("meteor")
    unknown = names - METRIC_OUTPUT_KEYS.keys()
    if unknown or not names:
        raise ValueError(f"Select metrics from {', '.join(METRIC_OUTPUT_KEYS)}; got {metrics!r}")
    return names


def max_num_refs(refs: List[List[str]]) -> int:
    return max((len(row_refs) for row_refs in refs), default=1)


def refs_by_position(refs: List[List[str]], num_refs: Optional[int] = None) -> List[List[Optional[str]]]:
    nrefs = num_refs or max_num_refs(refs)
    by_position: List[List[Optional[str]]] = []
    for idx in range(nrefs):
        by_position.append([
            row_refs[idx] if len(row_refs) > idx and row_refs[idx].strip() else None
            for row_refs in refs
        ])
    return by_position


def tokenize_text(text: str, language: str) -> str:
    if language == "ru":
        try:
            from razdel import tokenize

            return " ".join(token.text for token in tokenize(text))
        except Exception:
            return " ".join(text.split())
    if nltk is None:
        return " ".join(text.split())
    try:
        return " ".join(nltk.word_tokenize(text))
    except LookupError:
        return " ".join(text.split())


def simple_word_tokenize(text: str) -> List[str]:
    if nltk is not None:
        try:
            return nltk.word_tokenize(text.lower())
        except LookupError:
            pass
    return re.findall(r"\w+|[^\w\s]", text.lower(), flags=re.UNICODE)


def from_camel(text: str) -> str:
    text = text.replace("_", " ")
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    return text.lower()


def to_camel(text: str) -> str:
    parts = re.split(r"[\s_]+", text.strip())
    if not parts:
        return text
    return parts[0].lower() + "".join(part[:1].upper() + part[1:].lower() for part in parts[1:])


def linearize_predicate_for_factspotter(text: str) -> str:
    normalized = text.replace("_", " ").replace("-", " ").strip()
    pieces: List[str] = []
    for chunk in normalized.split():
        tokens = re.findall(r"[A-Z]+(?=[A-Z][a-z]|\d|\b)|[A-Z]?[a-z]+|\d+", chunk)
        if not tokens:
            tokens = [chunk]
        merged: List[str] = []
        index = 0
        while index < len(tokens):
            token = tokens[index]
            if (
                len(token) == 1
                and token.islower()
                and index + 1 < len(tokens)
                and tokens[index + 1].isupper()
                and len(tokens[index + 1]) > 1
            ):
                merged.append(token + tokens[index + 1])
                index += 2
                continue
            merged.append(token)
            index += 1
        pieces.extend(merged)
    return " ".join(piece.lower() for piece in pieces if piece)


def normalize_triple_parts(triple: str) -> Optional[Tuple[str, str, str]]:
    parts = [part.strip().strip('"') for part in str(triple).split(" | ")]
    if len(parts) != 3:
        parts = [part.strip().strip('"') for part in str(triple).split("|")]
    if len(parts) != 3 or not all(parts):
        return None
    return parts[0], parts[1], parts[2]


def clean_nli_object(text: str) -> str:
    return re.sub(r'^["\'](.*)["\']$', r"\1", text.strip())


def predicate_template_variants(predicate: str) -> List[str]:
    variants = []
    raw = predicate.strip()
    compact = raw.replace(" ", "_")
    spaced = from_camel(raw).replace("_", " ")
    camel = to_camel(spaced)
    for item in (
        raw,
        compact,
        spaced,
        camel,
        TEMPLATE_REMAP.get(raw),
        TEMPLATE_REMAP.get(compact),
        TEMPLATE_REMAP.get(spaced),
        TEMPLATE_REMAP.get(camel),
    ):
        if item and item not in variants:
            variants.append(item)
    return variants


def resolve_nli_template_file(metrics_dir: Path) -> Optional[Path]:
    candidates = [
        DEFAULT_NLI_TEMPLATE_FILE,
        metrics_dir.expanduser().resolve() / "nli_dusek" / "webnlg_templates.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


@lru_cache(maxsize=1)
def load_nli_model():
    try:
        import torch
        from transformers import RobertaForSequenceClassification, RobertaTokenizer
    except Exception:
        return None

    try:
        tokenizer = RobertaTokenizer.from_pretrained(DEFAULT_NLI_MODEL)
        model = RobertaForSequenceClassification.from_pretrained(DEFAULT_NLI_MODEL)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        return tokenizer, model, device
    except Exception:
        return None


@lru_cache(maxsize=8)
def load_nli_templates(template_path_str: str) -> Dict[str, Any]:
    template_path = Path(template_path_str)
    try:
        return json.loads(template_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def triple_to_nli_template(triple: str, templates: Dict[str, Any]) -> Optional[str]:
    parts = normalize_triple_parts(triple)
    if parts is None:
        return None
    subject, predicate, obj = parts

    template: Any = None
    for variant in predicate_template_variants(predicate):
        if variant in templates:
            template = templates[variant]
            break

    if template is None:
        template = f"The {from_camel(predicate)} of <subject> is <object>."

    if isinstance(template, dict):
        cleaned_obj = clean_nli_object(obj)
        template = template.get(obj) or template.get(cleaned_obj) or next(iter(template.values()))
    if isinstance(template, list):
        template = template[0] if template else ""

    template = str(template)
    template = template.replace("<subject>", subject)
    template = template.replace("<object>", clean_nli_object(obj))
    template = template.replace("_", " ")
    return template.strip()


def nli_entailment_probs(
    tokenizer: Any,
    model: Any,
    device: Any,
    pairs: List[Tuple[str, str]],
    batch_size: int = 8,
) -> List[float]:
    try:
        import torch
    except Exception:
        return []

    scores: List[float] = []
    with torch.no_grad():
        for start in progress_iter(
            range(0, len(pairs), batch_size),
            desc="NLI batches",
            unit="batch",
        ):
            batch = pairs[start:start + batch_size]
            premises = [item[0] for item in batch]
            hypotheses = [item[1] for item in batch]
            encoded = tokenizer(
                premises,
                hypotheses,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=1)[:, 2]
            scores.extend(float(item) for item in probs.detach().cpu().tolist())
    return scores


def tokenize_refs(refs: List[List[str]], language: str) -> List[List[str]]:
    return [[tokenize_text(ref, language) for ref in row_refs] for row_refs in refs]


def tokenize_hyps(hyps: List[str], language: str) -> List[str]:
    return [tokenize_text(hyp, language) for hyp in hyps]


def compute_toolkit_bleu(refs: List[List[str]], hyps: List[str], desc: str) -> float:
    refs_positions = refs_by_position(refs)
    bleu_metric = BLEU(effective_order=True)
    bleu_scores: List[float] = []
    for idx in progress_iter(range(len(hyps)), desc=desc):
        bleu_scores.append(
            bleu_metric.sentence_score(hyps[idx], [ref_list[idx] for ref_list in refs_positions]).score
        )
    return sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0


def load_toolkit_chrf(metrics_dir: Path):
    chrf_path = metrics_dir.expanduser().resolve() / "chrF.py"
    if not chrf_path.is_file():
        return None
    try:
        spec = importlib.util.spec_from_file_location("local_webnlg_chrf", chrf_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.computeChrF
    except Exception:
        return None


def compute_toolkit_chrf(refs: List[List[str]], hyps: List[str], metrics_dir: Path) -> float:
    compute_chrf = load_toolkit_chrf(metrics_dir)
    if compute_chrf is None:
        refs_for_sacrebleu = refs_by_position(refs)
        return CHRF(word_order=2, char_order=6, beta=2).corpus_score(hyps, refs_for_sacrebleu).score

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        hyp_path = tmpdir_path / "hypothesis_chrF"
        ref_path = tmpdir_path / "reference_chrF"
        filtered_refs: List[List[str]] = []
        filtered_hyps: List[str] = []
        for row_refs, hyp in zip(refs, hyps):
            valid_refs = [ref for ref in row_refs if ref.strip()]
            if valid_refs:
                filtered_refs.append(valid_refs)
                filtered_hyps.append(hyp)

        hyp_path.write_text("\n".join(filtered_hyps), encoding="utf-8")
        ref_path.write_text("\n".join("*#".join(row_refs) for row_refs in filtered_refs), encoding="utf-8")
        with ref_path.open("r", encoding="utf-8") as ref_handle, hyp_path.open("r", encoding="utf-8") as hyp_handle:
            total_f, _, _, _ = compute_chrf(ref_handle, hyp_handle, 2, 6, 2, None)
        return float(total_f)


def compute_toolkit_ter(refs: List[List[str]], hyps: List[str], language: str, desc: str) -> float:
    refs_tok = tokenize_refs(refs, language)
    hyps_tok = tokenize_hyps(hyps, language)
    if pyter is not None:
        ter_scores = []
        paired_items = list(zip(hyps_tok, refs_tok))
        for hyp, row_refs in progress_iter(paired_items, desc=desc):
            candidates = []
            for ref in row_refs:
                if not ref.strip():
                    candidates.append(1.0)
                else:
                    try:
                        candidates.append(pyter.ter(hyp.split(), ref.split()))
                    except Exception:
                        candidates.append(1.0)
            ter_scores.append(min(candidates) if candidates else 1.0)
        return sum(ter_scores) / len(ter_scores) if ter_scores else 0.0

    refs_for_sacrebleu = refs_by_position(refs_tok)
    return TER().corpus_score(hyps_tok, refs_for_sacrebleu).score / 100.0


def load_parent(metrics_dir: Path):
    parent_root = metrics_dir.expanduser().resolve() / "parent"
    if not parent_root.exists():
        return None
    parent_path = parent_root / "parent/parent.py"
    if not parent_path.is_file():
        return None
    if str(parent_root) not in sys.path:
        sys.path.insert(0, str(parent_root))
    try:
        from parent.parent import parent

        return parent
    except Exception:
        return None


def parent_table_from_triples(triples: List[str]) -> List[List[List[str]]]:
    table: List[List[List[str]]] = []
    for triple in triples:
        parts = [part.strip().strip('"') for part in triple.split(" | ")]
        if len(parts) != 3:
            continue
        subject, predicate, obj = parts
        table.append([
            simple_word_tokenize(subject.replace("_", " ")),
            simple_word_tokenize(to_camel(from_camel(predicate))),
            simple_word_tokenize(obj.replace("_", " ")),
        ])
    return table


def compute_parent_metric(
    refs: List[List[str]],
    hyps: List[str],
    graphs: List[List[str]],
    metrics_dir: Path,
) -> Optional[Dict[str, float]]:
    parent_fn = load_parent(metrics_dir)
    if parent_fn is None:
        return None
    predictions = [simple_word_tokenize(hyp) for hyp in hyps]
    references = [[simple_word_tokenize(ref) for ref in row_refs if ref.strip()] for row_refs in refs]
    tables = [parent_table_from_triples(graph) for graph in graphs]
    filtered = [
        (pred, row_refs, table)
        for pred, row_refs, table in zip(predictions, references, tables)
        if row_refs and table
    ]
    if not filtered:
        return None
    predictions, references, tables = map(list, zip(*filtered))
    n_jobs = min(8, os.cpu_count() or 1)
    precision, recall, f_score = parent_fn(
        predictions,
        references,
        tables,
        avg_results=True,
        n_jobs=n_jobs,
        use_tqdm=True,
    )
    return {
        "parent_precision": float(precision),
        "parent_recall": float(recall),
        "parent_f1": float(f_score),
    }


def compute_bertscore(refs: List[List[str]], hyps: List[str], language: str) -> Optional[Dict[str, float]]:
    try:
        from bert_score import score
    except Exception:
        return None
    filtered_refs = [[ref for ref in row_refs if ref.strip()] for row_refs in refs]
    try:
        precision, recall, f1 = score(hyps, filtered_refs, lang=language, verbose=True)
    except Exception:
        return None
    return {
        "bertscore_precision": float(precision.mean().item()),
        "bertscore_recall": float(recall.mean().item()),
        "bertscore_f1": float(f1.mean().item()),
    }


def compute_sbert_mpnet(refs: List[List[str]], hyps: List[str]) -> Optional[float]:
    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except Exception:
        return None

    flat_refs: List[str] = []
    counts: List[int] = []
    for row_refs in refs:
        valid_refs = [ref for ref in row_refs if ref.strip()]
        counts.append(len(valid_refs))
        flat_refs.extend(valid_refs)
    if not flat_refs:
        return None

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=device)
        hyp_embeddings = model.encode(
            hyps,
            batch_size=32,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=tqdm is not None,
        )
        ref_embeddings = model.encode(
            flat_refs,
            batch_size=32,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=tqdm is not None,
        )
    except Exception:
        return None

    scores: List[float] = []
    ref_index = 0
    for hyp_index, count in enumerate(counts):
        if not count:
            continue
        row_refs = ref_embeddings[ref_index:ref_index + count]
        similarities = torch.matmul(row_refs, hyp_embeddings[hyp_index])
        scores.append(float(torch.max(similarities).item()))
        ref_index += count
    return float(sum(scores) / len(scores)) if scores else None


def compute_metrics(
    refs: List[List[str]],
    hyps: List[str],
    graphs: List[List[str]],
    *,
    metrics: Set[str],
    meteor_jar: Path,
    language: str,
    metrics_dir: Path,
    meteor_disable_paraphrase: bool,
    log_prefix: str = "",
) -> Dict[str, float]:
    result: Dict[str, float] = {}

    if "bleu" in metrics:
        start = log_start(f"{log_prefix}BLEU")
        result["bleu"] = float(compute_toolkit_bleu(refs, hyps, desc=f"{log_prefix.strip()} BLEU samples".strip()))
        log_end(f"{log_prefix}BLEU", start)

    if "chrf++" in metrics:
        start = log_start(f"{log_prefix}chrF++")
        result["chrf++"] = to_percentage(compute_toolkit_chrf(refs, hyps, metrics_dir))
        log_end(f"{log_prefix}chrF++", start)

    if "rouge_l" in metrics:
        start = log_start(f"{log_prefix}ROUGE-L")
        rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        rouge_scores: List[float] = []
        for hyp, row_refs in progress_iter(list(zip(hyps, refs)), desc=f"{log_prefix.strip()} ROUGE-L samples".strip()):
            rouge_scores.append(max(rouge.score(ref, hyp)["rougeL"].fmeasure for ref in row_refs) * 100)
        result["rouge_l"] = float(sum(rouge_scores) / len(rouge_scores)) if rouge_scores else 0.0
        log_end(f"{log_prefix}ROUGE-L", start)

    if "meteor" in metrics:
        start = log_start(f"{log_prefix}METEOR")
        result["meteor"] = to_percentage(compute_meteor(refs, hyps, meteor_jar, language, meteor_disable_paraphrase))
        log_end(f"{log_prefix}METEOR", start)

    if "parent" in metrics:
        start = log_start(f"{log_prefix}PARENT")
        parent_scores = to_percentage_dict(compute_parent_metric(refs, hyps, graphs, metrics_dir))
        if parent_scores is not None:
            result.update(parent_scores)
        log_end(f"{log_prefix}PARENT", start)

    if "bertscore" in metrics:
        start = log_start(f"{log_prefix}BERTScore")
        bert_scores = to_percentage_dict(compute_bertscore(refs, hyps, language))
        if bert_scores is not None:
            result.update(bert_scores)
        log_end(f"{log_prefix}BERTScore", start)

    if "sbert" in metrics:
        start = log_start(f"{log_prefix}SBERT-MPNet")
        result["sbert_mpnet"] = to_percentage(compute_sbert_mpnet(refs, hyps))
        log_end(f"{log_prefix}SBERT-MPNet", start)

    required_keys = set().union(*(METRIC_OUTPUT_KEYS[metric] for metric in metrics))
    missing = sorted(
        key for key in required_keys
        if not isinstance(result.get(key), (int, float)) or not math.isfinite(result[key])
    )
    if missing:
        raise RuntimeError(
            f"Could not compute requested scores: {', '.join(missing)}. "
            "Check the metric dependencies and resources listed in repair_d2t/README.md."
        )
    return result


def original_metrics_cache_key(
    refs: List[List[str]],
    hyps: List[str],
    graphs: List[List[str]],
    *,
    metrics: Set[str],
    meteor_jar: Path,
    language: str,
    metrics_dir: Path,
    meteor_disable_paraphrase: bool,
) -> str:
    payload = {
        "cache_version": 2,
        "refs": refs,
        "hyps": hyps,
        "graphs": graphs,
        "metrics": sorted(metrics),
        "language": language,
        "meteor_jar": str(meteor_jar.expanduser().resolve()) if "meteor" in metrics else None,
        "metrics_dir": str(metrics_dir.expanduser().resolve()),
        "meteor_disable_paraphrase": meteor_disable_paraphrase,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.blake2b(raw.encode("utf-8"), digest_size=20).hexdigest()


def load_original_metrics_cache(cache_dir: Optional[Path], cache_key: str) -> Optional[Dict[str, Any]]:
    if cache_dir is None:
        return None
    cache_path = cache_dir / f"{cache_key}.json"
    if not cache_path.is_file():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if payload.get("cache_key") != cache_key or not isinstance(payload.get("metrics"), dict):
        return None
    print(f"  original cache hit {cache_key[:12]}", flush=True)
    return payload["metrics"]


def write_original_metrics_cache(cache_dir: Optional[Path], cache_key: str, metrics_value: Dict[str, Any]) -> None:
    if cache_dir is None:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{cache_key}.json"
    payload = {
        "cache_key": cache_key,
        "metrics": metrics_value,
    }
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  original cache write {cache_key[:12]}", flush=True)


def compute_original_metrics_with_cache(
    refs: List[List[str]],
    hyps: List[str],
    graphs: List[List[str]],
    *,
    metrics: Set[str],
    meteor_jar: Path,
    language: str,
    metrics_dir: Path,
    meteor_disable_paraphrase: bool,
    log_prefix: str,
    cache_dir: Optional[Path],
) -> Dict[str, Any]:
    cache_key = original_metrics_cache_key(
        refs,
        hyps,
        graphs,
        metrics=metrics,
        meteor_jar=meteor_jar,
        language=language,
        metrics_dir=metrics_dir,
        meteor_disable_paraphrase=meteor_disable_paraphrase,
    )
    cached = load_original_metrics_cache(cache_dir, cache_key)
    if cached is not None:
        return cached

    result = compute_metrics(
        refs,
        hyps,
        graphs,
        metrics=metrics,
        meteor_jar=meteor_jar,
        language=language,
        metrics_dir=metrics_dir,
        meteor_disable_paraphrase=meteor_disable_paraphrase,
        log_prefix=log_prefix,
    )
    write_original_metrics_cache(cache_dir, cache_key, result)
    return result


def compute_meteor(
    refs: List[List[str]],
    hyps: List[str],
    meteor_jar: Path,
    language: str,
    disable_paraphrase: bool,
) -> Optional[float]:
    if not meteor_jar.is_file() or shutil.which("java") is None:
        return None

    refs_tok = tokenize_refs(refs, language)
    hyps_tok = tokenize_hyps(hyps, language)
    num_refs = max_num_refs(refs_tok)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        hyp_path = tmpdir_path / "hypothesis_meteor"
        ref_path = tmpdir_path / "reference_meteor"
        hyp_path.write_text("\n".join(hyps_tok), encoding="utf-8")

        linear_references: List[str] = []
        for row_refs in refs_tok:
            padded_refs = row_refs + [""] * (num_refs - len(row_refs))
            linear_references.extend(padded_refs[:num_refs])
        ref_path.write_text("\n".join(linear_references), encoding="utf-8")

        lang = language if language in METEOR_LANGS else "other"
        cmd = [
            "java",
            "-Xmx2G",
            "-jar",
            str(meteor_jar),
            str(hyp_path),
            str(ref_path),
            "-l",
            lang,
        ]
        if lang != "other":
            cmd.append("-norm")
        cmd.extend(["-r", str(num_refs)])
        if disable_paraphrase:
            cmd.extend(["-m", "exact stem synonym", "-w", "1.0 0.6 0.8"])
        cmd.append("-q")

        try:
            result = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
            scores = [float(item) for item in result.decode("utf-8").split() if item.strip()]
            return scores[-1] if scores else None
        except Exception:
            return None


def evaluate_file(
    *,
    input_json: Path,
    output_dir: Path,
    metrics: Set[str],
    meteor_jar: Path,
    language: str,
    metrics_dir: Path,
    meteor_disable_paraphrase: bool,
    original_cache_dir: Optional[Path],
) -> Dict[str, Any]:
    file_start = time.perf_counter()
    print(f"[file] {input_json.name}", flush=True)
    rows = extract_rows(json.loads(ensure_file(input_json).read_text(encoding="utf-8")))
    original = collect_texts(rows, "input_text")
    repaired = collect_texts(rows, "repaired_text")
    graphs = trim_graphs_for_refs(collect_graphs(rows), len(original["refs"]))

    result = {
        "input_json": str(input_json),
        "metrics": sorted(metrics),
        "score_scale": "0-100",
        "num_samples": len(original["hyps"]),
        "original": compute_original_metrics_with_cache(
            original["refs"],
            original["hyps"],
            graphs,
            metrics=metrics,
            meteor_jar=meteor_jar,
            language=language,
            metrics_dir=metrics_dir,
            meteor_disable_paraphrase=meteor_disable_paraphrase,
            log_prefix="  original ",
            cache_dir=original_cache_dir,
        ),
        "repaired": compute_metrics(
            repaired["refs"],
            repaired["hyps"],
            graphs,
            metrics=metrics,
            meteor_jar=meteor_jar,
            language=language,
            metrics_dir=metrics_dir,
            meteor_disable_paraphrase=meteor_disable_paraphrase,
            log_prefix="  repaired ",
        ),
    }

    result["delta"] = {
        key: (
            result["repaired"][key] - result["original"][key]
            if result["original"].get(key) is not None and result["repaired"].get(key) is not None
            else None
        )
        for key in sorted(set(result["original"]) | set(result["repaired"]))
    }

    output_path = output_dir / f"{input_json.stem}_text_metrics.json"
    if output_path.is_file():
        try:
            previous = json.loads(output_path.read_text(encoding="utf-8"))
        except Exception:
            previous = None
        if isinstance(previous, dict):
            for section in ["original", "repaired"]:
                old_values = previous.get(section)
                if isinstance(old_values, dict):
                    old_values.update(result.get(section, {}))
                    result[section] = old_values
            previous_metrics = set(previous.get("metrics", []))
            result["metrics"] = sorted(previous_metrics | set(result["metrics"]))
            result["num_samples"] = previous.get("num_samples", result["num_samples"])
            result["score_scale"] = previous.get("score_scale", result["score_scale"])
            result["delta"] = {
                key: (
                    result["repaired"][key] - result["original"][key]
                    if result["original"].get(key) is not None and result["repaired"].get(key) is not None
                    else None
                )
                for key in sorted(set(result["original"]) | set(result["repaired"]))
            }

    output_dir.mkdir(parents=True, exist_ok=True)
    result["output_file"] = str(output_path)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[file] {input_json.name} total {time.perf_counter() - file_start:.2f}s", flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate original/repaired text metrics.")
    parser.add_argument("--input", type=Path, nargs="+", required=True, help="Repair full JSON files or dirs.")
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--metrics", type=str, default=DEFAULT_METRICS)
    parser.add_argument("--include_meteor", action="store_true", help="Backward-compatible alias for adding meteor.")
    parser.add_argument("--disable_original_cache", action="store_true")
    parser.add_argument("--original_cache_dir", type=Path, default=None)
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--metrics_dir", type=Path, default=DEFAULT_METRICS_DIR)
    parser.add_argument(
        "--meteor_jar",
        type=Path,
        default=DEFAULT_METEOR_JAR,
    )
    parser.add_argument(
        "--meteor_disable_paraphrase",
        action="store_true",
        help="Match METEOR 1.5 without paraphrase data when the full toolkit resources are unavailable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_files = collect_input_files(args.input)
    if not input_files:
        raise ValueError("No input JSON files found.")

    metrics = parse_metrics(args.metrics, args.include_meteor)
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else input_files[0].parent / "text_metrics"
    original_cache_dir = None
    if not args.disable_original_cache:
        original_cache_dir = (
            args.original_cache_dir.expanduser().resolve()
            if args.original_cache_dir
            else output_dir / ".cache/original_metrics"
        )
    iterator = tqdm(input_files, desc="Metric files", unit="file") if tqdm is not None else input_files
    summaries = [
        evaluate_file(
            input_json=input_json,
            output_dir=output_dir,
            metrics=metrics,
            meteor_jar=args.meteor_jar.expanduser().resolve(),
            language=args.language,
            metrics_dir=args.metrics_dir.expanduser().resolve(),
            meteor_disable_paraphrase=args.meteor_disable_paraphrase,
            original_cache_dir=original_cache_dir,
        )
        for input_json in iterator
    ]

    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
