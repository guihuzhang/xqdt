#!/usr/bin/env python3
"""Fresh end-to-end validation of T2D main-text Tables 9 and 10.

Replays compact traces for every main-text model/method/dataset/baseline at R1/R3,
runs the bundled WebNLG Text-to-RDF evaluator, compares Exact/Strict/Partial F1
against the bundled updated scores, and verifies rounded Exact F1 against
the paper-facing values documented in repair_t2d/README.md.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import tempfile
from pathlib import Path
from typing import Any

import reconstruct_compact_outputs as compact
import table_export_utils as tables

HERE = Path(__file__).resolve().parent
README = HERE / "README.md"
RESULTS = HERE / "results" / "repair_outputs"
INPUTS = HERE / "t2d_repair_input"
EVAL_MODULE_PATH = HERE / "eval" / "eval_text_to_data_repair.py"

METHODS = {
    "XQDT": ("xqdt", "g_bridge_recover_guard"),
    "SR": ("self_refine", "self_refine_v2"),
    "FS": ("factspotter", "factspotter_binary"),
    "NLI": ("nli", "nli_binary"),
    "PiVe": ("pive", "pive_missing"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tolerance", type=float, default=1e-12)
    return parser.parse_args()


def load_eval_module():
    spec = importlib.util.spec_from_file_location("t2d_eval", EVAL_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import evaluator from {EVAL_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def paper_exact_values() -> dict[tuple[str, str, str, str, str], float]:
    """Parse README Tables 9/10 into keyed Exact-F1 percentages."""
    text = README.read_text(encoding="utf-8")
    output: dict[tuple[str, str, str, str, str], float] = {}
    dataset_headers = {
        "### Table 9 / GenWiki": "genwiki",
        "### Table 10 / WebNLG": "webnlg",
    }
    for heading, dataset in dataset_headers.items():
        start = text.index(heading)
        tail = text[start:]
        next_heading = tail.find("\n### ", len(heading))
        block = tail if next_heading < 0 else tail[:next_heading]
        lines = [line for line in block.splitlines() if line.startswith("|")]
        data_lines = [line for line in lines if not line.startswith("| ---") and "Method" not in line]
        for line in data_lines:
            cells = [cell.strip().replace("**", "") for cell in line.strip("|").split("|")]
            method = cells[0]
            values = [float(value) for value in cells[1:]]
            columns = [
                ("bt5", "gemma3_12b"),
                ("bt5", "llama3_1_8b"),
                ("bt5", "qwen3_14b"),
                ("regen", "gemma3_12b"),
                ("regen", "llama3_1_8b"),
                ("regen", "qwen3_14b"),
            ]
            if method == "Baseline":
                for (baseline, model), value in zip(columns, values):
                    output[(dataset, baseline, model, "Baseline", "baseline")] = value
                continue
            match = re.fullmatch(r"(XQDT|SR|FS|NLI|PiVe)-R([13])", method)
            if not match:
                continue
            method_name, round_id = match.group(1), f"round{match.group(2)}"
            for (baseline, model), value in zip(columns, values):
                output[(dataset, baseline, model, method_name, round_id)] = value
    return output


def expected_summary(method: str, round_id: int, model: str, summaries: dict[str, dict]) -> tuple[dict, str]:
    if method == "xqdt":
        return summaries[f"xqdt_r{round_id}"], "g_bridge_recover_guard"
    if method == "self_refine":
        key = "baseline_r1" if round_id == 1 else "self_refine_r3"
        return summaries[key], "self_refine_v2"
    key = "baseline_r1" if round_id == 1 else "baseline_r3"
    summary = summaries[key]
    if round_id == 3:
        summary = tables.baseline_round_model_summary(key, model)
    return summary, compact.PROMPT_VARIANTS[method]


def reconstruct_states(method: str, model: str, dataset: str, baseline: str) -> tuple[list[str], dict[int, dict[str, list[str]]]]:
    unified_path, baseline_path = compact.input_paths(INPUTS, dataset, baseline)
    unified = read_json(unified_path)
    baseline_predictions = read_json(baseline_path)
    if len(unified) != len(baseline_predictions):
        raise ValueError(f"Input length mismatch for {dataset}/{baseline}")
    sample_ids = [str(row["sample_id"]) for row in unified]
    current = {
        sample_id: compact.normalize_baseline_predictions(baseline_predictions[idx])
        for idx, sample_id in enumerate(sample_ids)
    }
    records_path = compact.method_records_path(RESULTS, method, dataset)
    records = compact.load_records(records_path, dataset, baseline, model)
    states: dict[int, dict[str, list[str]]] = {0: {key: list(value) for key, value in current.items()}}
    cursor = 0
    for round_id in (1, 2, 3):
        while cursor < len(records) and int(records[cursor]["round"]) <= round_id:
            row = records[cursor]
            sample_id = str(row["sample_id"])
            if "triples" in row:
                current[sample_id] = [str(item) for item in row["triples"]]
            elif not bool(row.get("fell_back", False)):
                raise ValueError(f"Invalid sparse record: {method}/{model}/{dataset}/{baseline}/{sample_id}")
            cursor += 1
        states[round_id] = {key: list(value) for key, value in current.items()}
    return sample_ids, states


def raw_expected_f1(summary: dict, dataset: str, baseline: str, metric: str, *, model: str | None = None, variant: str | None = None, external: bool = False) -> float:
    group = summary["grouped"][dataset][baseline]
    if external:
        node = group["external"]
    else:
        node = tables.resolve_repair_entry(summary, dataset, baseline, model, variant)
    return float(node["total_scores"][metric]["F1"])


def main() -> None:
    args = parse_args()
    evalmod = load_eval_module()
    summaries = tables.load_all_summaries()
    paper = paper_exact_values()
    mismatches: list[str] = []
    fresh_eval_runs = 0
    metric_cells = 0
    paper_cells = 0

    with tempfile.TemporaryDirectory(prefix="xqdt-t2d-maintext-e2e-") as tmp_name:
        tmp = Path(tmp_name)

        # Score the four unique extractor baselines once each.
        for dataset in ("genwiki", "webnlg"):
            for baseline in ("bt5", "regen"):
                unified_path, baseline_path = compact.input_paths(INPUTS, dataset, baseline)
                unified = read_json(unified_path)
                baseline_predictions = read_json(baseline_path)
                sample_ids = [str(row["sample_id"]) for row in unified]
                state = {
                    sample_id: compact.normalize_baseline_predictions(baseline_predictions[idx])
                    for idx, sample_id in enumerate(sample_ids)
                }
                path = tmp / "outputs" / dataset / baseline / compact.baseline_filename(baseline, dataset)
                compact.write_json(
                    path,
                    compact.payload(
                        sample_ids,
                        state,
                        dataset=dataset,
                        baseline=baseline,
                        model=f"external_{baseline}",
                        prompt_variant="external_baseline",
                    ),
                )
                meta = {
                    "kind": "external",
                    "dataset": dataset,
                    "baseline": baseline,
                    "shot_k": 0,
                    "style": "markdown",
                    "label": f"external_{baseline}",
                    "repair_model": None,
                    "prompt_variant": None,
                }
                fresh = evalmod.evaluate_one(meta, path, tmp / "eval" / "baseline")
                fresh_eval_runs += 1
                for metric in ("Exact", "Strict", "Partial"):
                    actual = float(fresh["total_scores"][metric]["F1"])
                    expected = raw_expected_f1(summaries["baseline_r1"], dataset, baseline, metric, external=True)
                    metric_cells += 1
                    if abs(actual - expected) > args.tolerance:
                        mismatches.append(f"baseline/{dataset}/{baseline}/{metric}: fresh={actual:.12f} frozen={expected:.12f}")
                for model in tables.MAIN_TEXT_MODELS:
                    expected_paper = paper[(dataset, baseline, model, "Baseline", "baseline")]
                    actual_paper = round(100.0 * float(fresh["total_scores"]["Exact"]["F1"]), 2)
                    paper_cells += 1
                    if actual_paper != expected_paper:
                        mismatches.append(f"paper baseline/{dataset}/{baseline}/{model}: fresh={actual_paper:.2f} paper={expected_paper:.2f}")

        # Score every Table 9/10 repair cell at R1 and R3.
        for dataset in ("genwiki", "webnlg"):
            for baseline in ("bt5", "regen"):
                for model in tables.MAIN_TEXT_MODELS:
                    for paper_method, (method, variant) in METHODS.items():
                        sample_ids, states = reconstruct_states(method, model, dataset, baseline)
                        for round_id in (1, 3):
                            path = tmp / "outputs" / f"r{round_id}" / dataset / baseline / compact.output_filename(method, model, baseline, dataset)
                            compact.write_json(
                                path,
                                compact.payload(
                                    sample_ids,
                                    states[round_id],
                                    dataset=dataset,
                                    baseline=baseline,
                                    model=f"{model}_{baseline}_{variant}",
                                    prompt_variant=variant,
                                ),
                            )
                            meta = {
                                "kind": "repair",
                                "dataset": dataset,
                                "baseline": baseline,
                                "shot_k": 0,
                                "style": "markdown",
                                "label": f"{model}__{baseline}__{variant}",
                                "repair_model": model,
                                "prompt_variant": variant,
                            }
                            fresh = evalmod.evaluate_one(meta, path, tmp / "eval" / f"r{round_id}" / paper_method / model)
                            fresh_eval_runs += 1
                            expected_summary_obj, expected_variant = expected_summary(method, round_id, model, summaries)
                            for metric in ("Exact", "Strict", "Partial"):
                                actual = float(fresh["total_scores"][metric]["F1"])
                                expected = raw_expected_f1(
                                    expected_summary_obj,
                                    dataset,
                                    baseline,
                                    metric,
                                    model=model,
                                    variant=expected_variant,
                                )
                                metric_cells += 1
                                if abs(actual - expected) > args.tolerance:
                                    mismatches.append(
                                        f"{paper_method}/r{round_id}/{dataset}/{baseline}/{model}/{metric}: "
                                        f"fresh={actual:.12f} frozen={expected:.12f}"
                                    )
                            paper_round = f"round{round_id}"
                            expected_paper = paper[(dataset, baseline, model, paper_method, paper_round)]
                            actual_paper = round(100.0 * float(fresh["total_scores"]["Exact"]["F1"]), 2)
                            paper_cells += 1
                            if actual_paper != expected_paper:
                                mismatches.append(
                                    f"paper {paper_method}/r{round_id}/{dataset}/{baseline}/{model}: "
                                    f"fresh={actual_paper:.2f} paper={expected_paper:.2f}"
                                )

    if mismatches:
        print(
            f"[FAIL] fresh_eval_runs={fresh_eval_runs} metric_cells={metric_cells} "
            f"paper_cells={paper_cells} mismatches={len(mismatches)}"
        )
        for item in mismatches[:40]:
            print(item)
        raise SystemExit(1)

    print(
        f"[OK] fresh_eval_runs={fresh_eval_runs} metric_cells={metric_cells} "
        f"paper_cells={paper_cells} mismatches=0"
    )


if __name__ == "__main__":
    main()
