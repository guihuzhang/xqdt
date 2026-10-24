#!/usr/bin/env python3
import argparse
import json
import re
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import xml.etree.ElementTree as ET


REPAIR_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = Path(__file__).resolve().parent
EVAL_DIR = CODE_ROOT / "eval_webnlg_t2g"
EVAL_SCRIPT = EVAL_DIR / "Evaluation_script.py"
REFERENCE_XMLS = {
    "genwiki": CODE_ROOT / "data" / "genwiki_test_refs.xml",
    "webnlg": CODE_ROOT / "data" / "webnlg_test_refs.xml",
}
TOTAL_SCORE_KEYS = ("Ent_type", "Partial", "Strict", "Exact")
SUPPORTED_PROMPT_VARIANTS = (
    "r1_safe",
    "r2_scope_control",
    "r3_action_oriented",
    "g_align_local",
    "g_recover_if_clear",
    "g_guard_delete",
    "g_align_recover_guard",
    "g_bridge_recover_guard",
    "pive_missing",
    "self_refine",
    "self_refine_v2",
    "self_refine_v3",
    "nli_binary",
    "factspotter_binary",
    "self_verify_lite",
)
PROMPT_VARIANT_PATTERN = "|".join(re.escape(variant) for variant in SUPPORTED_PROMPT_VARIANTS)
EXTERNAL_RE = re.compile(
    r"^vllm_text_to_data_external_(?P<baseline>bt5|regen|fewshot)_(?P<dataset>genwiki|webnlg)_(?P<shot>\d+)shot_markdown$"
)
REPAIR_RE = re.compile(
    rf"^vllm_text_to_data_(?P<repair_model>.+)_(?P<baseline>bt5|regen|fewshot)_(?P<prompt_variant>{PROMPT_VARIANT_PATTERN})_(?P<dataset>genwiki|webnlg)_(?P<shot>\d+)shot_markdown$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate text_to_data_repair markdown outputs with the Hungarian T2G evaluator.")
    parser.add_argument(
        "--repair-root",
        type=Path,
        default=REPAIR_ROOT / "t2d_repair_output" / "xqdt_output" / "round1",
        help="Root directory containing markdown outputs and eval artifacts.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["genwiki", "webnlg"],
        choices=["genwiki", "webnlg"],
        help="Datasets to evaluate.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Directory for per-run hyps.xml and raw score outputs.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Path for the aggregated JSON summary.",
    )
    parser.add_argument(
        "--baselines",
        nargs="*",
        default=None,
        choices=["bt5", "regen", "fewshot"],
        help="Optional subset of baselines to evaluate.",
    )
    parser.add_argument(
        "--repair-models",
        nargs="*",
        default=None,
        help="Optional subset of repair model tags to evaluate, e.g. qwen3_8b llama3_1_8b.",
    )
    parser.add_argument(
        "--prompt-variants",
        nargs="*",
        default=None,
        choices=SUPPORTED_PROMPT_VARIANTS,
        help="Optional subset of repair prompt variants to evaluate.",
    )
    parser.add_argument(
        "--include-external",
        action="store_true",
        help="Also evaluate the external baseline markdown files stored in text_to_data_repair/markdown.",
    )
    args = parser.parse_args()
    if args.results_dir is None:
        args.results_dir = args.repair_root / "eval_results"
    if args.summary_json is None:
        args.summary_json = args.repair_root / "eval_summary_text_to_data_repair.json"
    return args


def ensure_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def load_output_file(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def triple_dict_to_string(item: Dict) -> str:
    subject = str(item.get("subject", "")).strip()
    predicate = str(item.get("predicate", "")).strip()
    obj = str(item.get("object", "")).strip()
    if not subject or not predicate or not obj:
        return ""
    return f"{subject} | {predicate} | {obj}"


def dedupe_preserve_order(triples: List[str]) -> List[str]:
    seen = set()
    deduped = []
    for triple in triples:
        if triple in seen:
            continue
        seen.add(triple)
        deduped.append(triple)
    return deduped


def sample_id_aliases(sample_id: str) -> List[str]:
    aliases = [sample_id]
    match = re.fullmatch(r"Id(\d+)", sample_id)
    if match:
        aliases.append(match.group(1))
    return aliases


def extract_predictions(payload: Dict) -> Dict[str, List[str]]:
    predictions: Dict[str, List[str]] = {}
    for entry in payload.get("entries", []):
        sample_id = str(entry.get("sample_id", "")).strip()
        if not sample_id:
            continue
        triples = []
        for triple_item in entry.get("triplets", []):
            if not isinstance(triple_item, dict):
                continue
            triple = triple_dict_to_string(triple_item)
            if triple:
                triples.append(triple)
        deduped = dedupe_preserve_order(triples)
        for alias in sample_id_aliases(sample_id):
            predictions[alias] = deduped
    return predictions


def build_hypothesis_xml(reference_xml: Path, predictions: Dict[str, List[str]], out_xml: Path) -> None:
    tree = ET.parse(reference_xml)
    root = tree.getroot()
    entries_parent = root.find("entries")
    if entries_parent is None:
        raise RuntimeError(f"invalid reference xml without <entries>: {reference_xml}")

    out_root = ET.Element("benchmark")
    out_entries = ET.SubElement(out_root, "entries")

    for ref_entry in entries_parent.findall("entry"):
        eid = ref_entry.attrib.get("eid", "")
        pred_entry = ET.SubElement(out_entries, "entry", attrib=deepcopy(ref_entry.attrib))
        generated = ET.SubElement(pred_entry, "generatedtripleset")
        for triple in predictions.get(eid, []):
            node = ET.SubElement(generated, "gtriple")
            node.text = triple

    xml_text = ET.tostring(out_root, encoding="unicode")
    out_xml.write_text('<?xml version="1.0" ?>\n' + xml_text, encoding="utf-8")


def parse_total_scores(score_text: str) -> Dict[str, Dict[str, float]]:
    scores: Dict[str, Dict[str, float]] = {}
    in_total = False
    current = None
    for raw_line in score_text.splitlines():
        line = raw_line.strip()
        if line == "Total scores":
            in_total = True
            continue
        if not in_total:
            continue
        if line == "Scores per tag":
            break
        if line in TOTAL_SCORE_KEYS:
            current = line
            scores[current] = {}
            continue
        if current is None:
            continue
        if line.startswith("Precision:"):
            match = re.search(r"Precision:\s*([0-9eE+\-\.]+)\s+Recall:\s*([0-9eE+\-\.]+)", line)
            if not match:
                raise RuntimeError(f"failed to parse precision/recall line: {line}")
            scores[current]["Precision"] = float(match.group(1))
            scores[current]["Recall"] = float(match.group(2))
            continue
        if line.startswith("F1:"):
            scores[current]["F1"] = float(line.split("F1:", 1)[1].strip())
            continue
    missing = [key for key in TOTAL_SCORE_KEYS if key not in scores or "F1" not in scores[key]]
    if missing:
        raise RuntimeError(f"failed to parse total scores for: {missing}")
    return scores


def parse_run_file(path: Path) -> Optional[Dict]:
    stem = path.stem
    match = EXTERNAL_RE.match(stem)
    if match:
        return {
            "kind": "external",
            "dataset": match.group("dataset"),
            "baseline": match.group("baseline"),
            "shot_k": int(match.group("shot")),
            "style": "markdown",
            "label": f"external_{match.group('baseline')}",
            "repair_model": None,
            "prompt_variant": None,
        }
    match = REPAIR_RE.match(stem)
    if match:
        return {
            "kind": "repair",
            "dataset": match.group("dataset"),
            "baseline": match.group("baseline"),
            "shot_k": int(match.group("shot")),
            "style": "markdown",
            "label": f"{match.group('repair_model')}__{match.group('baseline')}__{match.group('prompt_variant')}",
            "repair_model": match.group("repair_model"),
            "prompt_variant": match.group("prompt_variant"),
        }
    return None


def collect_runs(args: argparse.Namespace) -> List[Tuple[Dict, Path]]:
    selected_baselines = set(args.baselines) if args.baselines else None
    selected_models = set(args.repair_models) if args.repair_models else None
    selected_prompts = set(args.prompt_variants) if args.prompt_variants else None
    markdown_dirs = {
        "genwiki": args.repair_root / "markdown" / "genwiki",
        "webnlg": args.repair_root / "markdown" / "webnlg",
    }

    runs: List[Tuple[Dict, Path]] = []
    for dataset in args.datasets:
        markdown_dir = markdown_dirs[dataset]
        ensure_exists(markdown_dir, f"{dataset} repair markdown directory")
        for path in sorted(markdown_dir.glob("*.json")):
            meta = parse_run_file(path)
            if meta is None:
                continue
            if not args.include_external and meta["kind"] == "external":
                continue
            if selected_baselines and meta["baseline"] not in selected_baselines:
                continue
            if selected_models and meta["repair_model"] not in selected_models:
                continue
            if selected_prompts and meta["prompt_variant"] not in selected_prompts:
                continue
            runs.append((meta, path))
    return runs


def parse_model_sort_key(model_tag: str) -> Tuple[int, float, str]:
    family_order = {"gemma": 0, "llama": 1, "qwen": 2}
    family = next((name for name in family_order if model_tag.startswith(name)), "zzz")
    size = 9999.0
    size_match = re.search(r"(\d+)p(\d+)b|(\d+)b|(\d+)m", model_tag)
    if size_match:
        if size_match.group(1) and size_match.group(2):
            size = float(f"{size_match.group(1)}.{size_match.group(2)}")
        elif size_match.group(3):
            size = float(size_match.group(3))
        elif size_match.group(4):
            size = float(size_match.group(4)) / 1000.0
    return (family_order.get(family, 99), size, model_tag)


def sort_runs(items: List[Tuple[Dict, Path]]) -> List[Tuple[Dict, Path]]:
    prompt_order = {prompt: idx for idx, prompt in enumerate(SUPPORTED_PROMPT_VARIANTS)}
    baseline_order = {"bt5": 0, "regen": 1, "fewshot": 2}

    def key(item: Tuple[Dict, Path]) -> Tuple:
        meta, _ = item
        kind_rank = 0 if meta["kind"] == "external" else 1
        model_key = parse_model_sort_key(meta["repair_model"] or "external")
        return (
            meta["dataset"],
            baseline_order.get(meta["baseline"], 99),
            kind_rank,
            model_key,
            prompt_order.get(meta["prompt_variant"], 99),
            meta["label"],
        )

    return sorted(items, key=key)


def evaluate_one(meta: Dict, output_path: Path, results_dir: Path) -> Dict:
    payload = load_output_file(output_path)
    predictions = extract_predictions(payload)

    dataset = meta["dataset"]
    baseline = meta["baseline"]
    label = meta["label"]

    run_dir = results_dir / dataset / baseline / label
    run_dir.mkdir(parents=True, exist_ok=True)

    refs_xml = REFERENCE_XMLS[dataset]
    hyps_xml = run_dir / "hyps.xml"
    score_out = run_dir / "score.out"

    if score_out.exists():
        try:
            cached_score_text = score_out.read_text(encoding="utf-8")
            total_scores = parse_total_scores(cached_score_text)
            summary = {
                **meta,
                "output_path": str(output_path),
                "num_samples": len(payload.get("entries", [])),
                "refs_xml": str(refs_xml),
                "hyps_xml": str(hyps_xml),
                "score_out": str(score_out),
                "total_scores": total_scores,
            }
            (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[SKIP-EVAL] dataset={dataset} baseline={baseline} label={label} reuse={score_out}", flush=True)
            return summary
        except Exception:
            pass

    build_hypothesis_xml(refs_xml, predictions, hyps_xml)

    completed = subprocess.run(
        [sys.executable, str(EVAL_SCRIPT), str(refs_xml), str(hyps_xml)],
        cwd=str(EVAL_DIR),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"evaluator failed for {dataset}/{label} (exit={completed.returncode})\n"
            f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
        )
    score_out.write_text(completed.stdout, encoding="utf-8")
    total_scores = parse_total_scores(completed.stdout)
    summary = {
        **meta,
        "output_path": str(output_path),
        "num_samples": len(payload.get("entries", [])),
        "refs_xml": str(refs_xml),
        "hyps_xml": str(hyps_xml),
        "score_out": str(score_out),
        "total_scores": total_scores,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    ensure_exists(EVAL_SCRIPT, "Hungarian evaluator")
    for dataset in args.datasets:
        ensure_exists(REFERENCE_XMLS[dataset], f"{dataset} reference xml")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)

    runs = sort_runs(collect_runs(args))
    summary = {
        "evaluator": str(EVAL_SCRIPT),
        "reference_xmls": {dataset: str(path) for dataset, path in REFERENCE_XMLS.items() if dataset in args.datasets},
        "runs": [],
        "grouped": {dataset: {} for dataset in args.datasets},
    }

    for meta, output_path in runs:
        print(
            f"[EVAL] dataset={meta['dataset']} baseline={meta['baseline']} "
            f"kind={meta['kind']} label={meta['label']}",
            flush=True,
        )
        result = evaluate_one(meta, output_path, args.results_dir)
        summary["runs"].append(result)
        dataset_group = summary["grouped"].setdefault(meta["dataset"], {})
        baseline_group = dataset_group.setdefault(meta["baseline"], {"external": None, "repairs": {}})
        if meta["kind"] == "external":
            baseline_group["external"] = result
        else:
            repairs_group = baseline_group["repairs"].setdefault(meta["repair_model"], {})
            repairs_group[meta["prompt_variant"]] = result

    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] wrote summary json: {args.summary_json}")


if __name__ == "__main__":
    main()
