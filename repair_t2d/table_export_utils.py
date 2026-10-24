#!/usr/bin/env python3
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


REPAIR_ROOT = Path(__file__).resolve().parent
EXPORT_ROOT = REPAIR_ROOT / "exports"
UPDATED_SUMMARY = REPAIR_ROOT / "results" / "updated_scores.json"

SUMMARY_PATHS = {
    "xqdt_r1": REPAIR_ROOT / "t2d_repair_output" / "xqdt_output" / "round1" / "eval_summary_text_to_data_repair.json",
    "xqdt_r2": REPAIR_ROOT / "t2d_repair_output" / "xqdt_output" / "round2" / "eval_summary_text_to_data_repair.json",
    "xqdt_r3": REPAIR_ROOT / "t2d_repair_output" / "xqdt_output" / "round3" / "eval_summary_text_to_data_repair.json",
    "baseline_r1": REPAIR_ROOT / "t2d_repair_output" / "baseline" / "round1" / "eval_summary_text_to_data_repair.json",
    "baseline_r2": REPAIR_ROOT / "t2d_repair_output" / "baseline" / "round2_baseline" / "eval_summary_text_to_data_repair.json",
    "baseline_r3": REPAIR_ROOT / "t2d_repair_output" / "baseline" / "round3_baseline" / "eval_summary_text_to_data_repair.json",
    "self_refine_r2": REPAIR_ROOT / "t2d_repair_output" / "baseline" / "round2_self_refine" / "eval_summary_text_to_data_repair.json",
    "self_refine_r3": REPAIR_ROOT / "t2d_repair_output" / "baseline" / "round3_self_refine" / "eval_summary_text_to_data_repair.json",
}

DATASET_LABELS = {
    "genwiki": "GenWiki",
    "webnlg": "WebNLG",
}

BASELINE_LABELS = {
    "bt5": "BT5",
    "regen": "ReGen",
}

MODEL_LABELS = {
    "gemma3_4b": "G3-4B",
    "gemma3_12b": "G3-12B",
    "gemma3_27b": "G3-27B",
    "llama3_1_8b": "L3.1-8B",
    "llama3_3_70b_awq": "L3.3-70B",
    "qwen3_4b": "Q3-4B",
    "qwen3_8b": "Q3-8B",
    "qwen3_14b": "Q3-14B",
    "qwen3_32b": "Q3-32B",
}

MAIN_TEXT_MODELS = ["gemma3_12b", "llama3_1_8b", "qwen3_14b"]
APPENDIX_MODELS = [
    "gemma3_4b",
    "gemma3_12b",
    "gemma3_27b",
    "llama3_1_8b",
    "llama3_3_70b_awq",
    "qwen3_4b",
    "qwen3_8b",
    "qwen3_14b",
    "qwen3_32b",
]


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


@lru_cache(maxsize=None)
def load_json_cached(path_str: str) -> Dict:
    return load_json(Path(path_str))


@lru_cache(maxsize=4)
def load_all_summaries(summary_path: Path = UPDATED_SUMMARY) -> Dict[str, Dict]:
    bundle = load_json_cached(str(summary_path))
    profile = bundle["metadata"]["packages"]
    if profile["nervaluate"] != "0.2.0" or profile["nltk"] != "3.9.4":
        raise ValueError("The table bundle must use nervaluate 0.2.0 and nltk 3.9.4.")
    if len(bundle["runs"]) != 544:
        raise ValueError("The updated score bundle must contain all 544 evaluation runs.")
    summaries = {name: {"grouped": {}} for name in SUMMARY_PATHS}
    variants = {"xqdt": "g_bridge_recover_guard", "self_refine": "self_refine_v2",
                "factspotter": "factspotter_binary", "nli": "nli_binary", "pive": "pive_missing"}
    seen = set()
    for run in bundle["runs"]:
        key = (run["dataset"], run["baseline"], run["model"], run["method"], run["round"])
        if key in seen:
            raise ValueError(f"Duplicate result: {key}")
        seen.add(key)
        if run["round"] == 0:
            summary_key = "baseline_r1"
        elif run["method"] == "xqdt":
            summary_key = f'xqdt_r{run["round"]}'
        elif run["method"] == "self_refine" and run["round"] > 1:
            summary_key = f'self_refine_r{run["round"]}'
        else:
            summary_key = f'baseline_r{run["round"]}'
        group = summaries[summary_key]["grouped"].setdefault(run["dataset"], {}).setdefault(run["baseline"], {"repairs": {}})
        if run["round"] == 0:
            group["external"] = run
        else:
            group["repairs"].setdefault(run["model"], {})[variants[run["method"]]] = run
    return summaries


def resolve_repair_entry(
    summary: Dict,
    dataset: str,
    baseline: str,
    repair_model: str,
    prompt_variant: str,
) -> Dict:
    node = summary["grouped"][dataset][baseline]["repairs"][repair_model]
    if "total_scores" in node:
        return node
    return node[prompt_variant]


def metric_percent(
    summary: Dict,
    dataset: str,
    baseline: str,
    metric: str,
    *,
    repair_model: str | None = None,
    prompt_variant: str | None = None,
    external: bool = False,
) -> float:
    baseline_group = summary["grouped"][dataset][baseline]
    if external:
        node = baseline_group["external"]
    else:
        node = resolve_repair_entry(
            summary=summary,
            dataset=dataset,
            baseline=baseline,
            repair_model=repair_model,
            prompt_variant=prompt_variant,
        )
    return 100.0 * float(node["total_scores"][metric]["F1"])


def baseline_round_model_summary(round_key: str, repair_model: str) -> Dict:
    if round_key not in {"baseline_r2", "baseline_r3"}:
        raise ValueError(f"Unsupported round_key for model-level baseline summary: {round_key}")
    return load_all_summaries()[round_key]


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def format_value(value: float) -> str:
    return f"{value:.2f}"


def bold_column_maxima(rows: List[List[object]], start_col: int = 2, tol: float = 1e-9) -> List[List[str]]:
    if not rows:
        return []
    max_by_col = []
    for col_idx in range(start_col, len(rows[0])):
        max_by_col.append(max(float(row[col_idx]) for row in rows))

    formatted: List[List[str]] = []
    for row in rows:
        out = [str(row[0]), str(row[1])]
        for offset, col_idx in enumerate(range(start_col, len(row))):
            value = float(row[col_idx])
            rendered = format_value(value)
            if abs(value - max_by_col[offset]) <= tol:
                rendered = f"**{rendered}**"
            out.append(rendered)
        formatted.append(out)
    return formatted


def write_markdown(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def source_block(names: Iterable[str], summary_path: Path = UPDATED_SUMMARY) -> str:
    try:
        relative_path = summary_path.resolve().relative_to(REPAIR_ROOT.parent)
    except ValueError:
        relative_path = summary_path.name
    return f"## Source\n\n`{relative_path}`\n\nAll runs use nervaluate 0.2.0 and nltk 3.9.4."
