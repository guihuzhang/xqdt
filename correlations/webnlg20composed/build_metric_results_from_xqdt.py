#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = BASE_DIR / 'xqdt_results'
DEFAULT_OUTPUT_DIR = BASE_DIR / 'metric_results'


def require_dir(path: Path) -> Path:
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f'Required directory not found: {path}')
    return path


def compute_prf_from_errors(parsed_errors: Dict[str, List[str]], num_triples: int) -> Dict[str, float]:
    num_missing = len(parsed_errors['missing'])
    num_extra = len(parsed_errors['extra'])
    num_incorrect = len(parsed_errors['incorrect'])

    tp = max(0, num_triples - num_missing - num_incorrect)
    fp = num_extra + num_incorrect
    fn = num_missing + num_incorrect

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--triples-file', type=Path, help='Check prediction inputs against this triples file.')
    parser.add_argument(
        '--selected',
        nargs='*',
        default=None,
        help='Specific JSON file stems to convert, e.g. qwen3_8b_7479',
    )
    return parser.parse_args()


def iter_result_files(results_dir: Path, selected: Iterable[str] | None) -> List[Path]:
    selected_set = set(selected) if selected else None
    files: List[Path] = []
    for path in sorted(results_dir.glob('*.json')):
        if selected_set is not None and path.stem not in selected_set:
            continue
        files.append(path)
    return files


def load_results(path: Path) -> List[dict]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f'Expected list JSON in {path}, got {type(data).__name__}')
    if not data:
        raise ValueError(f'Empty result file: {path}')
    required_keys = {'team', 'entry_id', 'triples', 'parsed_errors'}
    missing = required_keys - set(data[0].keys())
    if missing:
        raise ValueError(f'{path} missing keys: {sorted(missing)}')
    return data


def build_team_scores(items: List[dict]) -> Dict[str, Dict[str, List[float]]]:
    by_team: Dict[str, Dict[int, Dict[str, float]]] = {}

    for item in items:
        team = item['team']
        entry_id = int(item['entry_id'])
        triples = item.get('triples', [])
        parsed_errors = item['parsed_errors']
        prf = compute_prf_from_errors(parsed_errors, len(triples))
        score_map = by_team.setdefault(team, {})
        if entry_id in score_map:
            raise ValueError(f'Duplicate result key: {(team, entry_id)}')
        score_map[entry_id] = prf

    team_scores: Dict[str, Dict[str, List[float]]] = {}
    for team, score_map in by_team.items():
        if not score_map:
            continue
        max_entry_id = max(score_map)
        precision = [''] * (max_entry_id + 1)
        recall = [''] * (max_entry_id + 1)
        f1 = [''] * (max_entry_id + 1)
        for entry_id, prf in score_map.items():
            precision[entry_id] = f'{prf["precision"]:.12g}'
            recall[entry_id] = f'{prf["recall"]:.12g}'
            f1[entry_id] = f'{prf["f1"]:.12g}'
        missing = [i for i, v in enumerate(precision) if v == '']
        unexpected_missing = [i for i in missing if i != 106]
        if unexpected_missing:
            raise ValueError(f'Team {team} has missing entry_ids: {unexpected_missing[:10]}')
        # The correlation loader skips internal entry 106.
        for i in missing:
            precision[i] = recall[i] = f1[i] = '0'
        team_scores[team] = {
            'p': precision,
            'r': recall,
            'f1': f1,
        }
    return team_scores


def write_metric_dir(output_dir: Path, folder_name: str, team_scores: Dict[str, Dict[str, List[float]]], score_key: str) -> None:
    folder = output_dir / folder_name
    folder.mkdir(parents=True, exist_ok=True)
    for team, scores in sorted(team_scores.items()):
        out_path = folder / f'{team}.txt'
        with open(out_path, 'w', encoding='utf-8') as f:
            for value in scores[score_key]:
                f.write(value + '\n')


def validate_input_coverage(items: List[dict], triples_file: Path | None = None) -> None:
    inputs_dir = BASE_DIR / 'webnlg2020_inputs'
    with (inputs_dir / 'triples.txt').open(encoding='utf-8') as handle:
        entry_count = sum(bool(line.strip()) for line in handle)
    team_files = sorted((inputs_dir / 'team-generations').glob('*.txt'))
    if not entry_count or not team_files:
        raise ValueError('Missing or empty WebNLG evaluation inputs')
    for path in team_files:
        with path.open(encoding='utf-8') as handle:
            if sum(1 for _ in handle) != entry_count:
                raise ValueError(f'Wrong number of text rows in {path}')
    expected = {(path.stem, entry_id) for path in team_files for entry_id in range(entry_count)}
    actual = [(item['team'], int(item['entry_id'])) for item in items]
    if len(actual) != len(set(actual)):
        raise ValueError('Duplicate result keys')
    evaluated = {key for key in expected if key[1] != 106}
    if set(actual) != expected and set(actual) != evaluated:
        raise ValueError('Result coverage does not match the WebNLG evaluation inputs')
    if triples_file is not None:
        graphs = [
            [triple.strip() for triple in line.split('<br>') if triple.strip()]
            for line in triples_file.read_text(encoding='utf-8').splitlines()
        ]
        if len(graphs) != entry_count:
            raise ValueError(f'{triples_file}: expected {entry_count} graphs, got {len(graphs)}')
        for item in items:
            if item['triples'] != graphs[int(item['entry_id'])]:
                raise ValueError(f"Input triples mismatch for {(item['team'], item['entry_id'])}")


def main() -> None:
    args = parse_args()
    results_dir = require_dir(args.results_dir)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    result_files = iter_result_files(results_dir, args.selected)
    if not result_files:
        raise FileNotFoundError(f'No result JSON files found under {results_dir}')

    for path in result_files:
        items = load_results(path)
        validate_input_coverage(items, args.triples_file)
        team_scores = build_team_scores(items)
        stem = path.stem
        print(f'Converting {stem} ({len(items)} items, {len(team_scores)} teams)')
        write_metric_dir(output_dir, f'dt_{stem}_p', team_scores, 'p')
        write_metric_dir(output_dir, f'dt_{stem}_r', team_scores, 'r')
        write_metric_dir(output_dir, f'dt_{stem}_f1', team_scores, 'f1')


if __name__ == '__main__':
    main()
