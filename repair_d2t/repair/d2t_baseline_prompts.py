"""Prompt builders used by the D2T Self-Refine, NLI, and FactSpotter repair baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence


@dataclass(frozen=True)
class Triple:
    subject: str
    predicate: str
    object: str


def format_triple(triple: Triple) -> str:
    return f"[S] {triple.subject} [P] {triple.predicate} [O] {triple.object}"


def format_triples(triples: Iterable[Triple]) -> str:
    return "\n".join(
        f"{idx}. {format_triple(triple)}" for idx, triple in enumerate(triples, start=1)
    )


def _join_lines(lines: Sequence[str]) -> str:
    return "\n".join(lines).strip() + "\n"


SELF_REFINE_SYSTEM = (
    "You are a factual revision assistant for data-to-text generation. "
    "First inspect the text against the source triples, then revise it."
)


def build_self_refine_markdown_prompt(
    source_triples: Sequence[Triple],
    previous_text: str,
) -> str:
    """Single-call self-refine prompt with a simple markdown table output.

    Design choices:
    - one round only
    - markdown table instead of JSON
    - no evidence column
    - missing/incorrect are represented as source triples
    - extra is represented as a short unsupported claim
    """

    return _join_lines(
        [
            "You are given source triples and a generated text.",
            "Inspect the text for factual issues, then output a revised text.",
            "",
            "Use exactly these issue types:",
            "- missing: a source triple is not expressed in the text.",
            "- incorrect: the text attempts to express a source triple but gets it wrong.",
            "- extra: the text contains a claim that is not supported by any source triple.",
            "",
            "Important constraints:",
            "- Be decisive but not noisy.",
            "- If the text is already factually faithful, keep it unchanged.",
            "- Do not add any fact that is not supported by the source triples.",
            "- You may rewrite locally or globally if needed, but revise only for factual faithfulness.",
            "",
            "Output exactly two sections in this order:",
            "1. ## Factual issues",
            "2. ## Revised text",
            "",
            "In ## Factual issues, output one markdown table with exactly these columns:",
            "| Type | Source triple or claim | Note |",
            "",
            "Formatting rules for the table:",
            "- For missing and incorrect, the second column must contain a source triple in this format:",
            "  [S] subject [P] predicate [O] object",
            "- For extra, the second column must contain a short unsupported claim copied or paraphrased from the text.",
            "- If there are no clear issues, output an empty table with only the header and separator rows.",
            "",
            "In ## Revised text, output only the final revised text.",
            "",
            "SOURCE TRIPLES:",
            format_triples(source_triples),
            "",
            "PREVIOUS TEXT:",
            previous_text.strip(),
        ]
    )


NLI_REVISION_SYSTEM = (
    "You are a factual revision assistant for data-to-text generation. "
    "Use NLI-derived feedback to revise the text."
)


def build_nli_revision_prompt(
    source_triples: Sequence[Triple],
    previous_text: str,
    facts_to_check: Sequence[Triple],
    global_extra_warning: Optional[str] = None,
) -> str:
    """Revision prompt for NLI feedback.

    The NLI signal is intentionally coarse:
    - facts_to_check: source facts not clearly supported or possibly mistranslated
    - global_extra_warning: one global hallucination warning from reverse NLI
    """

    feedback_lines = [
        "NLI feedback:",
        "- The listed source facts are not clearly supported in the text or may be expressed inaccurately.",
        "- This feedback is imperfect. Use it as revision guidance, not as a complete semantic parse.",
        "",
        "Source facts to check:",
    ]
    if facts_to_check:
        feedback_lines.extend(f"- {format_triple(triple)}" for triple in facts_to_check)
    else:
        feedback_lines.append("- none")

    if global_extra_warning:
        feedback_lines.extend(
            [
                "",
                "Global warning:",
                f"- {global_extra_warning}",
            ]
        )

    return _join_lines(
        [
            "Revise the previous text so that it faithfully expresses the source triples.",
            "You do not need to preserve the previous wording if rewriting is the cleanest way to restore factual faithfulness.",
            "Do not add facts outside the source triples.",
            "If the previous text is already faithful, you may keep it unchanged.",
            "",
            *feedback_lines,
            "",
            "Output only the final revised text.",
            "",
            "SOURCE TRIPLES:",
            format_triples(source_triples),
            "",
            "PREVIOUS TEXT:",
            previous_text.strip(),
        ]
    )


FACTSPOTTER_REVISION_SYSTEM = (
    "You are a factual revision assistant for data-to-text generation. "
    "Use FactSpotter-style source-fact support feedback to revise the text."
)


def build_factspotter_revision_prompt(
    source_triples: Sequence[Triple],
    previous_text: str,
    facts_to_check: Sequence[Triple],
) -> str:
    """Revision prompt for FactSpotter feedback.

    This prompt deliberately omits an extra-claim channel. FactSpotter is used
    here only as a source-fact support signal.
    """

    feedback_lines = [
        "FactSpotter feedback:",
        "- The listed source facts received low factual-support confidence with respect to the text.",
        "- Treat them as the main places to inspect and repair.",
        "",
        "Source facts to check:",
    ]
    if facts_to_check:
        feedback_lines.extend(f"- {format_triple(triple)}" for triple in facts_to_check)
    else:
        feedback_lines.append("- none")

    return _join_lines(
        [
            "Revise the previous text so that it faithfully expresses the source triples.",
            "You may rewrite freely if needed, but the final text must stay within the source triples.",
            "If the previous text is already faithful, you may keep it unchanged.",
            "",
            *feedback_lines,
            "",
            "Output only the final revised text.",
            "",
            "SOURCE TRIPLES:",
            format_triples(source_triples),
            "",
            "PREVIOUS TEXT:",
            previous_text.strip(),
        ]
    )

