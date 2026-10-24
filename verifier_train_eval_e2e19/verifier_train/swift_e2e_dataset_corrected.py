#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Triple Error Detection Dataset for Swift 3.x
Clean version - JSON uses 'input_text' instead of 'text'
"""
import random
from pathlib import Path
import spacy
from typing import Dict, Any, List, Tuple
from swift.llm import ResponsePreprocessor, DatasetMeta, register_dataset, load_dataset

nlp = spacy.load("en_core_web_sm")


def format_triple(triple_str: str) -> str:
    """ Format a single triple to [S] subject [P] predicate [O] object
    Returns: Formatted triple string """
    parts = [p.strip() for p in triple_str.split('|')]
    if len(parts) == 3:
        return f"[S] {parts[0]} [P] {parts[1]} [O] {parts[2]}"
    return triple_str


def shuffle_text_sentences(text: str) -> str:
    """
    Shuffle sentences in the text using spacy for accurate sentence splitting
    Args:
        text: Input text
    Returns:
        Text with shuffled sentences
    """
    # Use spacy to split into sentences
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    if len(sentences) <= 1:
        return text
    # Shuffle sentences
    random.shuffle(sentences)
    # Join with space
    return ' '.join(sentences)


def get_entity_position_in_text(triple_str: str, text: str) -> int:
    """
    Get the earliest position where any entity from the triple appears in text
    Returns:
        Position of first entity appearance (or large number if not found)
    """
    parts = [p.strip() for p in triple_str.split('|')]
    if len(parts) != 3:
        return 999999

    subject, predicate, obj = parts
    text_lower = text.lower()

    positions = []
    if subject.lower() in text_lower:
        positions.append(text_lower.find(subject.lower()))
    if obj.lower() in text_lower:
        positions.append(text_lower.find(obj.lower()))

    return sum(positions) / len(positions) if positions else 999999


def find_best_matching_position(error_triple: str, triple_list: List[str]) -> int:
    """
    Find the best matching triple position in the list
    Strategy: find triple with maximum number of matching components (S/P/O)
    Returns:
        Position in list (1-indexed), or 999999 if not found
    """
    error_parts = [p.strip() for p in error_triple.split('|')]
    if len(error_parts) != 3:
        return 999999
    best_match_pos = 999999
    best_match_score = 0
    for i, triple in enumerate(triple_list, start=1):
        parts = [p.strip() for p in triple.split('|')]
        if len(parts) != 3:
            continue
        # Count how many components match
        score = sum(1 for e, t in zip(error_parts, parts) if e.lower() == t.lower())
        # Prefer earlier positions if scores are equal
        if score > best_match_score:
            best_match_score = score
            best_match_pos = i

    return best_match_pos


def build_query(text: str, triples: List[str]) -> str:
    """
    Build query prompt with text and triples

    Args:
        text: Input text
        triples: List of triples (shuffled or not)

    Returns:
        Query string for model
    """
    triples_formatted = '\n'.join(
        f'{i}. {format_triple(t)}' for i, t in enumerate(triples, start=1)
    )

    query = f"""Verify if the triples align with the text. Find missing, extra, or incorrect triples.
TEXT: {text}
TRIPLES:
{triples_formatted}
Output as markdown table with Type and Triple columns."""

    return query


def build_response(row: Dict, label: str, text: str, shuffled_triples: List[str]) -> str:
    """
    Build response in markdown table format
    Returns:
        Markdown table response or "All correct"
    """
    if label == 'positive':
        return "All correct"

    # Use 'or []' to handle both missing keys and None values
    extra = row.get('extra_contents') or []
    missing = row.get('missing_triples') or []
    incorrect_mappings = row.get('incorrect_mappings') or []

    if not extra and not missing and not incorrect_mappings:
        return "All correct"

    table_lines = ["| Type | Triple |", "| ---- | ------ |"]

    # Missing triples - sorted by entity position in (possibly shuffled) text
    missing_items = [
        (get_entity_position_in_text(t, text), format_triple(t))
        for t in missing
        if len([p.strip() for p in t.split('|')]) == 3
    ]
    missing_items.sort()
    for _, formatted in missing_items:
        table_lines.append(f"| Missing | {formatted} |")

    # Extra triples - sorted by position in shuffled list
    extra_items = [
        (find_best_matching_position(t, shuffled_triples), format_triple(t))
        for t in extra
        if len([p.strip() for p in t.split('|')]) == 3
    ]
    extra_items.sort()
    for _, formatted in extra_items:
        table_lines.append(f"| Extra | {formatted} |")

    # Incorrect triples - sorted by position in shuffled list
    incorrect_items = [
        (find_best_matching_position(m.get('incorrect', ''), shuffled_triples),
         format_triple(m.get('incorrect', '')))
        for m in incorrect_mappings
        if m.get('incorrect') and len([p.strip() for p in m.get('incorrect', '').split('|')]) == 3
    ]
    incorrect_items.sort()
    for _, formatted in incorrect_items:
        table_lines.append(f"| Incorrect | {formatted} |")

    return '\n'.join(table_lines)


class TripleErrorPreprocessor(ResponsePreprocessor):
    """Basic preprocessor for triple error detection"""

    def __init__(
            self,
            shuffle_triples: bool = True,
            shuffle_text_prob: float = 0.0,
            seed: int = None
    ):
        """
        Initialize preprocessor

        Args:
            shuffle_triples: Whether to shuffle triples
            shuffle_text_prob: Probability of shuffling text sentences (0.0 to 1.0)
            seed: Random seed for reproducibility
        """
        super().__init__()
        self.shuffle_triples = shuffle_triples
        self.shuffle_text_prob = shuffle_text_prob
        if seed is not None:
            random.seed(seed)

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        text = row.get('input_text', '')
        triples = row.get('triples', [])
        label = row.get('label', 'positive')

        # print(f"\n--- ORIGINAL TEXT ---")
        # print(text)
        #
        # print(f"\n--- ORIGINAL TRIPLES ---")
        # for i, t in enumerate(triples, start=1):
        #     print(f"{i}. {format_triple(t)}")
        #
        # # Print error information if negative sample
        # if label == 'negative':
        #     extra = row.get('extra_triples') or []
        #     missing = row.get('missing_triples') or []
        #     incorrect_mappings = row.get('incorrect_mappings') or []
        #
        #     if extra:
        #         print(f"\n--- EXTRA TRIPLES (errors) ---")
        #         for t in extra:
        #             print(f"  {format_triple(t)}")
        #
        #     if missing:
        #         print(f"\n--- MISSING TRIPLES (errors) ---")
        #         for t in missing:
        #             print(f"  {format_triple(t)}")
        #
        #     if incorrect_mappings:
        #         print(f"\n--- INCORRECT TRIPLES (errors) ---")
        #         for m in incorrect_mappings:
        #             print(f"  Original:  {format_triple(m.get('original', ''))}")
        #             print(f"  Incorrect: {format_triple(m.get('incorrect', ''))}")

        if self.shuffle_text_prob > 0 and random.random() < self.shuffle_text_prob:
            text = shuffle_text_sentences(text)
            # print(f"\n--- SHUFFLED TEXT ---")
            # print(text)
        shuffled_triples = triples.copy()
        if self.shuffle_triples:
            random.shuffle(shuffled_triples)
            # print(f"\n--- SHUFFLED TRIPLES ---")
            # for i, t in enumerate(shuffled_triples, start=1):
            #     print(f"{i}. {format_triple(t)}")
        query = build_query(text, shuffled_triples)
        response = build_response(row, label, text, shuffled_triples)

        # print(f"\n--- FINAL RESPONSE ---")
        # print(response)

        return super().preprocess({
            'query': query,
            'response': response
        })


def register_triple_error_dataset(
        train_json: str = str(Path(__file__).resolve().parents[1] / "data" / "train_joint_e2e_webnlg_all_samples.json"),
        val_json: str = str(Path(__file__).resolve().parents[1] / "data" / "dev_joint_e2e_webnlg_all_samples.json"),
        test_json: str = str(Path(__file__).resolve().parents[1] / "data" / "test_all_samples.json"),
        shuffle_triples_train: bool = True,
        shuffle_triples_val: bool = False,
        shuffle_triples_test: bool = False,
        shuffle_text_prob_train: float = 0.3,
        shuffle_text_prob_val: float = 0.0,
        shuffle_text_prob_test: float = 0.0,
        seed: int = None
):
    """
    Register triple error detection datasets to Swift

    Args:
        train_json: Training data path
        val_json: Validation data path
        test_json: Test data path
        shuffle_triples_train: Whether to shuffle triples in training data
        shuffle_triples_val: Whether to shuffle triples in validation data
        shuffle_triples_test: Whether to shuffle triples in test data
        shuffle_text_prob_train: Probability of shuffling text in training (0.0-1.0)
        shuffle_text_prob_val: Probability of shuffling text in validation (0.0-1.0)
        shuffle_text_prob_test: Probability of shuffling text in test (0.0-1.0)
        seed: Random seed for reproducibility
    """
    train_preprocessor = TripleErrorPreprocessor(
        shuffle_triples=shuffle_triples_train,
        shuffle_text_prob=shuffle_text_prob_train,
        seed=seed
    )
    val_preprocessor = TripleErrorPreprocessor(
        shuffle_triples=shuffle_triples_val,
        shuffle_text_prob=shuffle_text_prob_val,
        seed=seed
    )
    test_preprocessor = TripleErrorPreprocessor(
        shuffle_triples=shuffle_triples_test,
        shuffle_text_prob=shuffle_text_prob_test,
        seed=seed
    )

    register_dataset(DatasetMeta(
        dataset_name='triple_error_train',
        dataset_path=train_json,
        preprocess_func=train_preprocessor
    ))

    register_dataset(DatasetMeta(
        dataset_name='triple_error_val',
        dataset_path=val_json,
        preprocess_func=val_preprocessor
    ))

    register_dataset(DatasetMeta(
        dataset_name='triple_error_test',
        dataset_path=test_json,
        preprocess_func=test_preprocessor
    ))

    print("Registered datasets:")
    print(f"  - triple_error_train: {train_json}")
    print(f"    shuffle_triples={shuffle_triples_train}, shuffle_text_prob={shuffle_text_prob_train}")
    print(f"  - triple_error_val: {val_json}")
    print(f"    shuffle_triples={shuffle_triples_val}, shuffle_text_prob={shuffle_text_prob_val}")
    print(f"  - triple_error_test: {test_json}")
    print(f"    shuffle_triples={shuffle_triples_test}, shuffle_text_prob={shuffle_text_prob_test}")


def test_dataset():
    """Test dataset loading and basic functionality"""
    print("\nTesting Triple Error Dataset with Spacy")
    print("=" * 60)

    # Register with shuffling enabled
    register_triple_error_dataset(
        shuffle_triples_train=True,
        shuffle_triples_val=True,
        shuffle_triples_test=False,
        shuffle_text_prob_train=0.5,
        shuffle_text_prob_val=0.5,
        shuffle_text_prob_test=0.5,
        seed=42
    )

    print("\nLoading datasets...")
    val_dataset = load_dataset('triple_error_val')[0]

    print(f"Val: {len(val_dataset)} samples")

    if len(val_dataset) > 0:
        print("First validation sample:")
        for sample in val_dataset:
            if 'messages' in sample:
                messages = sample['messages']

                if len(messages) >= 1:
                    query = messages[0]['content']
                    print(f"\n--- QUERY ---")
                    print(query)

                if len(messages) >= 2:
                    response = messages[1]['content']
                    print(f"\n--- RESPONSE ---")
                    print(response)

    print("\n" + "=" * 60)
    print("Test completed!")


if __name__ == '__main__':
    test_dataset()
