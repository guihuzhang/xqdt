#!/usr/bin/env python3
import argparse
import json
import re
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
TEXT_TO_DATA_ROOT = ROOT / "generations" / "text_to_data"
CODE_ROOT = Path(__file__).resolve().parent
EVAL_DIR = CODE_ROOT / "eval_webnlg_t2g"
EVAL_SCRIPT = EVAL_DIR / "Evaluation_script.py"
REFERENCE_XMLS = {
    "genwiki": CODE_ROOT / "data" / "genwiki_test_refs.xml",
    "webnlg": CODE_ROOT / "data" / "webnlg_test_refs.xml",
}
OUTPUT_STYLE_DIRS = {
    "json": TEXT_TO_DATA_ROOT / "json",
    "markdown": TEXT_TO_DATA_ROOT / "markdown",
}
TOTAL_SCORE_KEYS = ("Ent_type", "Partial", "Strict", "Exact")
FILE_RE = re.compile(
    r"^vllm_text_to_data_(?P<model_tag>.+)_(?P<dataset>genwiki|webnlg)_(?P<shot>\d+)shot_(?P<style>json|markdown)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate text_to_data extraction outputs with the Hungarian T2G evaluator.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["genwiki", "webnlg"],
        choices=["genwiki", "webnlg"],
        help="Datasets to evaluate.",
    )
    parser.add_argument(
        "--styles",
        nargs="+",
        default=["json", "markdown"],
        choices=["json", "markdown"],
        help="Output styles to evaluate.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=TEXT_TO_DATA_ROOT / "eval_results",
        help="Directory for per-run hyps.xml and raw score outputs.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=TEXT_TO_DATA_ROOT / "eval_summary_text_to_data.json",
        help="Path for the aggregated JSON summary.",
    )
    parser.add_argument(
        "--summary-md",
        type=Path,
        default=TEXT_TO_DATA_ROOT / "eval_summary_text_to_data.md",
        help="Path for the markdown comparison table.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional subset of model tags to evaluate.",
    )
    return parser.parse_args()


def ensure_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def load_output_file(path: Path) -> Dict:
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if path.suffix == ".jsonl":
        entries = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return {
            "dataset": path.parent.name,
            "output_style": "unknown",
            "shot_k": None,
            "prompt_style": None,
            "model": None,
            "entries": entries,
        }
    raise ValueError(f"unsupported file type: {path}")


def collect_runs(datasets: Iterable[str], styles: Iterable[str]) -> Dict[str, Dict[str, Dict[str, Path]]]:
    runs: Dict[str, Dict[str, Dict[str, Path]]] = {dataset: {style: {} for style in styles} for dataset in datasets}
    for style in styles:
        for dataset in datasets:
            style_dir = OUTPUT_STYLE_DIRS[style] / dataset
            ensure_exists(style_dir, f"{style}/{dataset} output directory")
            for path in sorted(style_dir.glob("*.json")):
                match = FILE_RE.match(path.stem)
                if not match:
                    continue
                model_tag = match.group("model_tag")
                runs[dataset][style][model_tag] = path
    return runs


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


def evaluate_one(output_path: Path, dataset: str, style: str, results_dir: Path) -> Dict:
    payload = load_output_file(output_path)
    model_name = payload.get("model")
    predictions = extract_predictions(payload)

    match = FILE_RE.match(output_path.stem)
    if not match:
        raise RuntimeError(f"unexpected file name format: {output_path.name}")
    model_tag = match.group("model_tag")

    run_dir = results_dir / dataset / style / model_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    refs_xml = REFERENCE_XMLS[dataset]
    hyps_xml = run_dir / "hyps.xml"
    score_out = run_dir / "score.out"
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
            f"evaluator failed for {dataset}/{style}/{model_tag} (exit={completed.returncode})\n"
            f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
        )
    score_out.write_text(completed.stdout, encoding="utf-8")
    total_scores = parse_total_scores(completed.stdout)
    summary = {
        "dataset": dataset,
        "style": style,
        "model_tag": model_tag,
        "model_name": model_name,
        "num_samples": len(payload.get("entries", [])),
        "refs_xml": str(refs_xml),
        "hyps_xml": str(hyps_xml),
        "score_out": str(score_out),
        "total_scores": total_scores,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


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


def display_model_name(model_tag: str) -> str:
    return model_tag.replace("_nothink", "")


def build_markdown(summary: Dict) -> str:
    lines = [
        "# Text-to-Data Extraction Eval",
        "",
        f"- Evaluator: `{EVAL_SCRIPT}`",
        "- Matching: Hungarian version (`Evaluation_script.py`)",
        "- Python: uses the active environment's `python` by default",
        "- Compared styles: `json` vs `markdown`",
        "- Metrics shown below are Total-score F1 from the official T2G evaluator",
        "",
    ]

    for dataset in ("genwiki", "webnlg"):
        dataset_block = summary["datasets"].get(dataset, {})
        json_models = dataset_block.get("json", {})
        md_models = dataset_block.get("markdown", {})
        lines.extend([f"## {dataset}", ""])
        shared_models = sorted(set(json_models) & set(md_models), key=parse_model_sort_key)
        lines.extend(
            [
                "| Model | Style | Exact | Strict | Partial | Ent_type |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for model_tag in shared_models:
            json_scores = json_models[model_tag]["total_scores"]
            md_scores = md_models[model_tag]["total_scores"]
            json_row = [
                display_model_name(model_tag),
                "json",
                f'{json_scores["Exact"]["F1"]:.4f}',
                f'{json_scores["Strict"]["F1"]:.4f}',
                f'{json_scores["Partial"]["F1"]:.4f}',
                f'{json_scores["Ent_type"]["F1"]:.4f}',
            ]
            md_row = [
                display_model_name(model_tag),
                "markdown",
                f'{md_scores["Exact"]["F1"]:.4f}',
                f'{md_scores["Strict"]["F1"]:.4f}',
                f'{md_scores["Partial"]["F1"]:.4f}',
                f'{md_scores["Ent_type"]["F1"]:.4f}',
            ]
            lines.append("| " + " | ".join(json_row) + " |")
            lines.append("| " + " | ".join(md_row) + " |")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    ensure_exists(EVAL_SCRIPT, "Hungarian evaluator")
    for dataset in args.datasets:
        ensure_exists(REFERENCE_XMLS[dataset], f"{dataset} reference xml")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_md.parent.mkdir(parents=True, exist_ok=True)

    runs = collect_runs(args.datasets, args.styles)
    selected_models = set(args.models) if args.models else None

    summary = {
        "evaluator": str(EVAL_SCRIPT),
        "reference_xmls": {dataset: str(path) for dataset, path in REFERENCE_XMLS.items() if dataset in args.datasets},
        "datasets": {dataset: {style: {} for style in args.styles} for dataset in args.datasets},
    }

    for dataset in args.datasets:
        for style in args.styles:
            for model_tag, output_path in sorted(runs[dataset][style].items(), key=lambda item: parse_model_sort_key(item[0])):
                if selected_models and model_tag not in selected_models:
                    continue
                print(f"[EVAL] dataset={dataset} style={style} model={model_tag}", flush=True)
                result = evaluate_one(output_path, dataset, style, args.results_dir)
                summary["datasets"][dataset][style][model_tag] = result

    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    args.summary_md.write_text(build_markdown(summary), encoding="utf-8")
    print(f"[DONE] wrote summary json: {args.summary_json}")
    print(f"[DONE] wrote summary md  : {args.summary_md}")


if __name__ == "__main__":
    main()
