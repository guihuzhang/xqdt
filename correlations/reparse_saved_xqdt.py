#!/usr/bin/env python3
"""Reparse saved fine-tuned XQDT responses without running a model."""

import argparse
import copy
import json
import re
from pathlib import Path
from typing import Dict, List


def parse_model_response(response: str) -> Dict[str, List[str]]:
    result = {'missing': [], 'extra': [], 'incorrect': []}
    if not isinstance(response, str) or not response.strip():
        raise ValueError("Empty model response")
    response = re.sub(r'<think>\s*</think>', '', response).strip()
    for line in response.strip().splitlines():
        if '|' not in line:
            continue
        parts = [part.strip() for part in line.strip().strip('|').split('|')]
        if len(parts) < 2:
            continue
        lowered = [part.strip('* ').lower() for part in parts]
        if lowered[0] == 'type' and lowered[1].startswith('triple'):
            continue
        if all(re.fullmatch(r':?[-=]{3,}:?', part) for part in parts):
            continue
        error_type = lowered[0]
        triple = ' '.join(parts[1:]).strip()
        labels = {
            'missing': 'missing', 'omission': 'missing', 'omitted': 'missing',
            'extra': 'extra', 'additional': 'extra',
            'incorrect': 'incorrect', 'wrong': 'incorrect', 'error': 'incorrect',
        }
        if error_type not in labels:
            raise ValueError(f"Unrecognized error-table row: {line}")
        if not triple:
            raise ValueError(f"Missing triple in error-table row: {line}")
        result[labels[error_type]].append(triple)
    if any(result.values()):
        return result
    positive = re.fullmatch(
        r'(?:all (?:triples (?:are )?)?correct|no errors?(?: (?:found|detected))?|'
        r'(?:the triples (?:and (?:the )?text )?are )?fully aligned)[.!]?',
        response.strip().strip('* ').lower(),
    )
    if positive:
        return result
    raise ValueError("Response contains neither explicit all-correct text nor error rows")


def parse_webnlg17_response(item):
    response = re.sub(r'<think>\s*</think>', '', item['model_response']).strip()
    shorthand = re.fullmatch(
        r'all (?:(?:rows|triples) (?:are )?)?(incorrect|missing)[.!]?',
        response.strip('* ').lower(),
    )
    if shorthand:
        triples = item['formatted_triples']
        if not triples or len(triples) != len(item['triples']):
            raise ValueError('Expected the complete input triples for an all-error response')
        parsed = {'missing': [], 'extra': [], 'incorrect': []}
        parsed[shorthand.group(1)] = list(triples)
        return parsed
    response = re.sub(
        r'<pre>.*?</pre>', lambda match: ' '.join(match.group().split()),
        response, flags=re.DOTALL | re.IGNORECASE,
    )
    response = '\n'.join(
        line for line in response.splitlines()
        if not re.fullmatch(r'\s*\|?\s*text\s*\|\s*null\s*\|?\s*', line, re.IGNORECASE)
    )
    return parse_model_response(response)


def refresh_item(item, dataset):
    updated = copy.deepcopy(item)
    if dataset in {"webnlg17", "webnlg20"} and item.get("prompt") == "Skipped":
        if item["text"].strip() or item["model_response"] != "Empty text - all triples missing":
            raise ValueError("Unrecognized skipped response")
        return updated
    parsed = (parse_webnlg17_response(item) if dataset == "webnlg17"
              else parse_model_response(item["model_response"]))
    counts = {label: len(values) for label, values in parsed.items()}
    updated["parsed_errors"] = parsed
    updated["error_counts"] = counts.copy()
    if dataset in {"webnlg17", "webnlg20"}:
        updated["error_counts"]["total"] = sum(counts.values())
        updated["is_aligned"] = not any(counts.values())
        return updated
    total = int(item["num_triples"])
    missing, extra, incorrect = counts["missing"], counts["extra"], counts["incorrect"]
    correct = max(0, total - missing - incorrect)
    false_positive, false_negative = extra + incorrect, missing + incorrect
    precision = correct / (correct + false_positive) if correct + false_positive else 0.0
    recall = correct / (correct + false_negative) if correct + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    updated.update(precision=precision, recall=recall, f1=f1,
                   TP=correct, FP=false_positive, FN=false_negative)
    if "prf_fixed" in updated:
        updated["prf_fixed"] = dict(
            num_missing=missing, num_extra=extra, num_incorrect=incorrect,
            num_correct=correct, denom_precision=correct + false_positive,
            denom_recall=correct + false_negative)
    return updated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["4lrp", "webnlg17", "webnlg20"], required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.results_dir.resolve() == args.output_dir.resolve():
        raise ValueError("Choose a separate output directory to keep the source responses")
    pattern = "*_results.json" if args.dataset == "4lrp" else "*.json"
    paths = sorted(args.results_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No {pattern} files in {args.results_dir}")
    prepared = []
    for path in paths:
        output = args.output_dir / path.name
        if output.exists():
            raise FileExistsError(f"Output already exists: {output}")
        items = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(items, list) or not items:
            raise ValueError(f"Expected a nonempty result list: {path}")
        keys = [(item["team"], item["entry_id"]) if args.dataset != "4lrp"
                else item["id"] for item in items]
        if len(set(keys)) != len(keys):
            raise ValueError(f"Duplicate sample keys: {path}")
        updated = []
        for item in items:
            try:
                updated.append(refresh_item(item, args.dataset))
            except (KeyError, ValueError, TypeError) as error:
                raise ValueError(f"{path.name}, sample {item.get('id')}: {error}") from error
        changed = sum(before != after for before, after in zip(items, updated))
        prepared.append((output, updated, changed))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for output, items, changed in prepared:
        output.write_text(json.dumps(items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"{output.name}: {len(items)} samples, {changed} updated")


if __name__ == "__main__":
    main()
