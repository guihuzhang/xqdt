"""Prompt variants for first-round repair ablation.

These variants keep the same high-level section layout as the current repair
prompt, but each variant provides a complete section-level configuration rather
than only appending extra rules.
"""

from __future__ import annotations

from typing import Any, Dict

BASE_REPAIR_SYSTEM = (
    "You are a data-to-text revision assistant. Preserve correct content while "
    "correcting flagged errors."
)

DEFAULT_REPAIR_PROMPT_VARIANT = "p1c_p2b"


PROMPT_VARIANTS: Dict[str, Dict[str, Any]] = {
    "p0_original": {
        "label": "P0 Original",
        "intent": "Current baseline repair prompt.",
        "repair_system": BASE_REPAIR_SYSTEM,
        "task_lines": [
            "Produce a complete revised text that is a faithful lexicalisation of the source triples.",
            "The source triples are the only facts allowed in the output.",
            "The verifier feedback is not a complete list of all facts. It only highlights issues to fix.",
            "Preserve all source-supported content from the previous text unless it conflicts with the source triples or feedback.",
        ],
        "missing_guidance": (
            "Add these facts only if they are not already expressed in the previous text.\n"
            "Integrate them naturally into the existing text.\n"
            "Do not duplicate an existing fact with different wording."
        ),
        "extra_guidance": (
            "Remove only the unsupported information listed below.\n"
            "Preserve the rest of the sentence if it is source-supported.\n"
            "Do not remove source-supported facts from the same sentence."
        ),
        "incorrect_guidance": (
            "For each listed fact, find the corresponding claim in the previous text. "
            "The listed subject, predicate, and object are the correct source fact.\n"
            "If the previous text conflicts with any part of the listed subject, predicate, and object, "
            "make the smallest local correction so the final text naturally expresses the correct fact.\n"
            "If the previous text already expresses the fact faithfully, including by clear implication, keep the wording.\n"
            "Do not turn natural wording into awkward predicate-style phrases.\n"
            "Do not remove other source-supported facts from the same sentence."
        ),
        "revision_requirements": [
            "Output the complete revised text, not just the edits.",
            "Keep all correct source-supported information from the previous text.",
            "Address the flagged issues according to the guidance above, without shortening the text to only the flagged fact.",
            "Use only facts supported by the source triples.",
            "Do not mention the verifier, feedback, or triples in the final text.",
            'Output only one valid JSON object with exactly this schema: {"full-text": "..."}',
        ],
    },
    "p1a_validate_preserve": {
        "label": "P1A Validate / Preserve",
        "intent": "Reduce false-alarm-driven edits and explicitly preserve already-correct graph facts.",
        "repair_system": BASE_REPAIR_SYSTEM,
        "task_lines": [
            "Produce a complete revised text that is a faithful lexicalisation of the source triples.",
            "The source triples are the only facts allowed in the output.",
            "Use the verifier feedback as guidance to inspect possible issues, but confirm each issue against the source triples and the previous text before revising.",
            "The verifier feedback only points to possible issues to inspect; it is not a complete account of all source-supported content that should remain in the text.",
            "If the previous text already faithfully expresses a fact supported by the source triples, keep it unchanged unless it conflicts with another source-supported fact.",
            "Preserve all source-supported content from the previous text unless it conflicts with the source triples.",
        ],
        "missing_guidance": (
            "Add a listed fact only when it is not already faithfully expressed in the previous text.\n"
            "If the fact is already clearly conveyed, keep the existing wording.\n"
            "When adding a genuinely missing fact, integrate it naturally instead of restating an existing fact."
        ),
        "extra_guidance": (
            "Remove only information that is unsupported by the source triples after checking the previous text carefully.\n"
            "If the previous text already gives a source-supported meaning with different wording, keep it.\n"
            "Do not remove or alter other source-supported content that is not part of the issue being fixed."
        ),
        "incorrect_guidance": (
            "For each listed fact, compare the previous text against the source triples and identify the local claim that may need revision.\n"
            "If the previous text already expresses the source fact faithfully, do not rewrite it only to mirror the predicate label more literally.\n"
            "If the previous text truly conflicts with the source fact, revise only what is needed so the final wording naturally expresses the correct fact.\n"
            "Do not remove or alter other source-supported content that is not part of the issue being fixed."
        ),
        "revision_requirements": [
            "Output the complete revised text, not just the edits.",
            "If the previous text already faithfully expresses the source facts, output it unchanged.",
            "Keep all correct source-supported information from the previous text.",
            "Fix the flagged issues while preserving the surrounding source-supported content; do not collapse the revision into a text that only states the flagged fact.",
            "Use only facts supported by the source triples.",
            "Do not mention the verifier, feedback, or triples in the final text.",
            'Output only one valid JSON object with exactly this schema: {"full-text": "..."}',
        ],
    },
    "p1b_validate_preserve": {
        "label": "P1B Validate / Preserve",
        "intent": "Same P1 policy, phrased as an explicit inspect-then-decide procedure.",
        "repair_system": BASE_REPAIR_SYSTEM,
        "task_lines": [
            "Produce a complete revised text that is a faithful lexicalisation of the source triples.",
            "The source triples are the only facts allowed in the output.",
            "Use the verifier feedback to inspect possible issues, then check each highlighted issue against the source triples and the previous text before deciding whether any revision is necessary.",
            "The verifier feedback only identifies possible problem areas; it does not list all source-supported content that should remain in the final text.",
            "Keep any wording that already faithfully expresses a fact supported by the source triples, unless it conflicts with another source-supported fact.",
            "Revise only when the previous text fails to faithfully express a source-supported fact or includes unsupported content.",
        ],
        "missing_guidance": (
            "For each listed missing fact, first check whether that fact is already faithfully expressed in the previous text.\n"
            "If it is already expressed, keep the existing wording.\n"
            "If it is genuinely missing, add it naturally without duplicating facts that are already present."
        ),
        "extra_guidance": (
            "For each listed extra fact, verify that it is truly unsupported by the source triples.\n"
            "If the wording differs but the meaning is source-supported, keep it.\n"
            "Remove only the unsupported meaning and preserve all other source-supported content."
        ),
        "incorrect_guidance": (
            "For each listed incorrect fact, compare the previous text with the source triples and determine whether the relevant claim is actually inconsistent.\n"
            "If the previous text already expresses the source fact faithfully, keep it.\n"
            "If it truly conflicts with the source fact, revise only what is needed so the final wording naturally expresses the correct fact.\n"
            "Do not remove or alter other source-supported content that is not part of the issue being fixed."
        ),
        "revision_requirements": [
            "Output the complete revised text, not just the edits.",
            "If the previous text already faithfully expresses the source facts, output it unchanged.",
            "Keep all correct source-supported information from the previous text.",
            "Fix the flagged issues while preserving the surrounding source-supported content; do not collapse the revision into a text that only states the flagged fact.",
            "Use only facts supported by the source triples.",
            "Do not mention the verifier, feedback, or triples in the final text.",
            'Output only one valid JSON object with exactly this schema: {"full-text": "..."}',
        ],
    },
    "p1c_validate_preserve": {
        "label": "P1C Validate / Preserve",
        "intent": "Same P1 policy, phrased with more explicit if/then decision rules.",
        "repair_system": BASE_REPAIR_SYSTEM,
        "task_lines": [
            "Produce a complete revised text that is a faithful lexicalisation of the source triples.",
            "The source triples are the only facts allowed in the output.",
            "Use the verifier feedback as a signal to inspect the previous text, but decide whether to revise by checking the source triples and the previous text together.",
            "The verifier feedback points to possible issues, not to all source-supported content that should remain in the text.",
            "If the previous text already faithfully expresses a source-supported fact, keep that wording unchanged.",
            "If the previous text conflicts with a source-supported fact or contains unsupported content, revise only the conflicting content.",
        ],
        "missing_guidance": (
            "If a listed missing fact is already faithfully expressed in the previous text, do not add it again.\n"
            "If it is not expressed, add it naturally.\n"
            "Do not restate an already-correct fact just to match the verifier wording."
        ),
        "extra_guidance": (
            "If a listed extra fact is truly unsupported by the source triples, remove it.\n"
            "If the wording is different but the meaning is source-supported, keep it.\n"
            "Do not remove or alter other source-supported content that is not part of the issue being fixed."
        ),
        "incorrect_guidance": (
            "If the previous text already expresses the source fact faithfully, do not rewrite it only to mirror the predicate label more literally.\n"
            "If the previous text truly conflicts with the source fact, correct that claim so the final wording faithfully expresses the source fact.\n"
            "Do not remove or alter other source-supported content that is not part of the issue being fixed."
        ),
        "revision_requirements": [
            "Output the complete revised text, not just the edits.",
            "If the previous text already faithfully expresses the source facts, output it unchanged.",
            "Keep all correct source-supported information from the previous text.",
            "Fix the flagged issues while preserving the surrounding source-supported content; do not collapse the revision into a text that only states the flagged fact.",
            "Use only facts supported by the source triples.",
            "Do not mention the verifier, feedback, or triples in the final text.",
            'Output only one valid JSON object with exactly this schema: {"full-text": "..."}',
        ],
    },
    "p2_scope_control": {
        "label": "P2 Scope Control",
        "intent": "Constrain edit scope and distinguish local repair from broader repair.",
        "repair_system": BASE_REPAIR_SYSTEM,
        "task_lines": [
            "Produce a complete revised text that is a faithful lexicalisation of the source triples.",
            "The source triples are the only facts allowed in the output.",
            "Use the verifier feedback as guidance to inspect possible issues, but confirm each issue against the source triples and the previous text before revising.",
            "If the previous text already faithfully expresses a fact supported by the source triples, keep it unchanged unless it conflicts with another source-supported fact.",
            "Prefer the smallest revision scope that fully fixes the issue.",
            "If the issue is local, edit only the affected span or clause. If the wording around an issue is broadly broken, recover only the affected region of text rather than rewriting unrelated content.",
        ],
        "missing_guidance": (
            "Add a listed fact only when it is not already faithfully expressed in the previous text.\n"
            "Integrate it into the affected local region when possible.\n"
            "Do not expand unrelated content just to patch one missing fact."
        ),
        "extra_guidance": (
            "Remove only the unsupported part needed to fix the issue.\n"
            "Avoid changing unaffected content outside the local span needed to fix the issue.\n"
            "Preserve nearby source-supported material unless a broader local rewrite is clearly necessary."
        ),
        "incorrect_guidance": (
            "For each listed fact, locate the specific part of the previous text that may conflict with the source triples.\n"
            "If the previous text already expresses the source fact faithfully, do not rewrite it only to mirror the predicate label more literally.\n"
            "When a correction is needed, make the smallest local change that restores factual correctness and natural wording.\n"
            "Avoid changing unaffected content outside the local span needed to fix the issue."
        ),
        "revision_requirements": [
            "Output the complete revised text, not just the edits.",
            "If the previous text already faithfully expresses the source facts, output it unchanged.",
            "Keep all correct source-supported information from the previous text.",
            "If the previous text contains a broken wrapper such as ERROR or malformed JSON fragments, first recover the intended natural-language content from it, then revise the facts. Still return the final answer in the required JSON format.",
            "Use only facts supported by the source triples.",
            "Do not mention the verifier, feedback, or triples in the final text.",
            'Output only one valid JSON object with exactly this schema: {"full-text": "..."}',
        ],
    },
    "p3_surface_preserve_and_merge": {
        "label": "P3 Surface Preservation + Merge",
        "intent": "Protect exact strings and discourage additive patching.",
        "repair_system": BASE_REPAIR_SYSTEM,
        "task_lines": [
            "Produce a complete revised text that is a faithful lexicalisation of the source triples.",
            "The source triples are the only facts allowed in the output.",
            "Use the verifier feedback as guidance to inspect possible issues, but confirm each issue against the source triples and the previous text before revising.",
            "If the previous text already faithfully expresses a fact supported by the source triples, keep it unchanged unless it conflicts with another source-supported fact.",
            "Prefer the smallest revision scope that fully fixes the issue.",
            "Protect names, dates, URLs, identifiers, formulas, and parenthetical qualifiers unless the source triples explicitly require changing them.",
        ],
        "missing_guidance": (
            "Add a listed fact only when it is not already faithfully expressed in the previous text.\n"
            "When adding a genuinely missing fact, merge it into the existing sentence or clause whenever possible.\n"
            "Do not append a lazy repair fragment if the fact can be integrated naturally."
        ),
        "extra_guidance": (
            "Remove only the unsupported content that must be removed.\n"
            "Preserve exact strings that remain source-supported, including names, URLs, formulas, titles, and parenthetical qualifiers.\n"
            "Avoid changing unaffected content outside the local span needed to fix the issue."
        ),
        "incorrect_guidance": (
            "For each listed fact, locate the specific conflicting span in the previous text.\n"
            "If the previous text already expresses the source fact faithfully, do not rewrite it only to mirror the predicate label more literally.\n"
            "When a correction is needed, preserve any exact source-supported strings around the local fix unless the source triples explicitly require changing them.\n"
            "Prefer natural English relation realization over ontology-like predicate wording."
        ),
        "revision_requirements": [
            "Output the complete revised text, not just the edits.",
            "If the previous text already faithfully expresses the source facts, output it unchanged.",
            "Keep all correct source-supported information from the previous text.",
            "Preserve source-supported exact strings such as URLs, formulas, identifiers, titles, and parenthetical qualifiers unless a factual correction requires changing them.",
            "Do not use additive patch patterns such as 'Additionally' or 'Also' unless a new sentence is structurally necessary.",
            "Use only facts supported by the source triples.",
            "Do not mention the verifier, feedback, or triples in the final text.",
            'Output only one valid JSON object with exactly this schema: {"full-text": "..."}',
        ],
    },
    "u1_router_balanced": {
        "label": "U1 Router Balanced",
        "intent": "Unified router prompt: no-op when already faithful, local patch when possible, sentence rewrite only when clearly needed.",
        "repair_system": BASE_REPAIR_SYSTEM,
        "task_lines": [
            "Produce a complete revised text that is a faithful lexicalisation of the source triples.",
            "The source triples are the only facts allowed in the output.",
            "Use the verifier feedback as a signal to inspect the previous text, but decide whether to revise by checking the source triples and the previous text together.",
            "Before revising, first decide whether any change is strictly necessary.",
            "If the previous text already faithfully expresses the source-supported facts, keep it unchanged.",
            "If a single local edit is enough, keep the revision local to the affected span or clause.",
            "If several linked problems occur in the same sentence, or a local patch would leave the sentence awkward or still misleading, rewrite only the affected sentence or affected local region.",
        ],
        "missing_guidance": (
            "First check whether each listed fact is already faithfully expressed in the previous text.\n"
            "If it is already expressed, do not add it again.\n"
            "If it is genuinely missing and can be inserted locally, add it with a local insertion.\n"
            "If a local insertion would make the sentence awkward or incomplete, rewrite only the affected sentence or affected local region.\n"
            "Do not add a fact merely to mirror the verifier wording more literally."
        ),
        "extra_guidance": (
            "Remove only information that is truly unsupported by the source triples.\n"
            "If the unsupported meaning is confined to a local span, delete or replace only that span.\n"
            "If the unsupported meaning makes the whole sentence locally broken, rewrite only the affected sentence or affected local region.\n"
            "Preserve all other source-supported content."
        ),
        "incorrect_guidance": (
            "For each listed incorrect fact, first locate the exact conflicting span in the previous text.\n"
            "If no exact conflicting span can be identified and the previous text already conveys the source fact faithfully, keep the wording unchanged.\n"
            "If a single local correction is enough, patch only that clause or span.\n"
            "If multiple linked facts in the same sentence are wrong, or a local patch would remain awkward or misleading, rewrite only the affected sentence or affected local region.\n"
            "Do not rewrite a sentence only to make it look closer to the triple format."
        ),
        "revision_requirements": [
            "Output the complete revised text, not just the edits.",
            "If no change is strictly necessary, output the previous text unchanged.",
            "Keep all correct source-supported information from the previous text.",
            "Prefer local edits when they are sufficient, but prefer factual correctness when a larger sentence-level repair is clearly needed.",
            "Use only facts supported by the source triples.",
            "Do not mention the verifier, feedback, or triples in the final text.",
            'Output only one valid JSON object with exactly this schema: {"full-text": "..."}',
        ],
    },
    "u2_structured_to_nl_bridge": {
        "label": "U2 Structured-to-NL Bridge",
        "intent": "Keep structured verifier signals, but translate them into natural-language repair instructions instead of mirroring triple structure literally.",
        "repair_system": BASE_REPAIR_SYSTEM,
        "task_lines": [
            "Produce a complete revised text that is a faithful lexicalisation of the source triples.",
            "The source triples are the only facts allowed in the output.",
            "Use the verifier feedback as a signal to inspect the previous text, but decide whether to revise by checking the source triples and the previous text together.",
            "Treat each highlighted fact as a fact to check in the previous text, not as wording to copy literally.",
            "If the previous text already faithfully expresses a source-supported fact in natural language, keep that wording unchanged.",
            "If revision is needed, prefer the smallest change that makes the text faithfully express the source fact.",
        ],
        "missing_guidance": (
            "For each listed missing fact, first check whether the previous text already conveys that fact, even if the wording is different.\n"
            "If it already does, keep the wording unchanged.\n"
            "If it does not, add the fact naturally to the relevant sentence or clause.\n"
            "Do not restate a fact only to match the verifier wording more literally."
        ),
        "extra_guidance": (
            "For each listed extra fact, determine whether the previous text truly expresses unsupported content.\n"
            "If the meaning is actually supported by the source triples, keep it.\n"
            "If the meaning is unsupported, remove or locally revise only the span that expresses the unsupported content.\n"
            "Preserve all other source-supported content."
        ),
        "incorrect_guidance": (
            "For each listed incorrect fact, first identify what claim in the previous text should be checked.\n"
            "Ask whether the previous text already expresses the same source fact correctly in natural language.\n"
            "If it already does, keep the wording.\n"
            "If it conflicts with the source fact, revise only the clause or local region that expresses that relation.\n"
            "Do not rewrite a sentence merely to make it resemble the triple structure more closely."
        ),
        "revision_requirements": [
            "Output the complete revised text, not just the edits.",
            "If the previous text already faithfully expresses the source facts, output it unchanged.",
            "Keep all correct source-supported information from the previous text.",
            "Use the feedback to decide what to check, but not as a template for literal predicate-style wording.",
            "Use only facts supported by the source triples.",
            "Do not mention the verifier, feedback, or triples in the final text.",
            'Output only one valid JSON object with exactly this schema: {"full-text": "..."}',
        ],
    },
    "u3_router_hardfix_surface_guard": {
        "label": "U3 Router Hard-Fix + Surface Guard",
        "intent": "Unified router prompt with stronger factual correction plus surface-form guardrails for titles, names, and other exact strings.",
        "repair_system": BASE_REPAIR_SYSTEM,
        "task_lines": [
            "Produce a complete revised text that is a faithful lexicalisation of the source triples.",
            "The source triples are the only facts allowed in the output.",
            "Use the verifier feedback as a signal to inspect the previous text, but decide whether to revise by checking the source triples and the previous text together.",
            "Before revising, first decide whether any change is strictly necessary.",
            "If the previous text already faithfully expresses the source-supported facts, keep it unchanged.",
            "If a factual conflict is clear, you must correct it.",
            "If a single local edit is enough, keep the revision local to the affected span or clause.",
            "If several linked problems occur in the same sentence, or a local patch would leave the sentence awkward or still misleading, rewrite only the affected sentence or affected local region.",
            "Preserve natural surface forms of names, titles, years, identifiers, formulas, and parenthetical qualifiers unless they are directly involved in the factual error.",
        ],
        "missing_guidance": (
            "First check whether each listed fact is already faithfully expressed in the previous text.\n"
            "If it is already expressed, do not add it again.\n"
            "If it is genuinely missing, add it.\n"
            "Prefer a local insertion when it is enough, but rewrite the affected sentence or affected local region when that is necessary to express the fact naturally and correctly.\n"
            "Do not replace a natural title or name mention with a more canonical form unless the original mention is factually wrong."
        ),
        "extra_guidance": (
            "Remove only information that is truly unsupported by the source triples.\n"
            "If the unsupported meaning is confined to a local span, delete or replace only that span.\n"
            "If the unsupported meaning makes the whole sentence locally broken, rewrite only the affected sentence or affected local region.\n"
            "Preserve unaffected source-supported names, titles, years, identifiers, formulas, and parenthetical qualifiers."
        ),
        "incorrect_guidance": (
            "For each listed incorrect fact, first locate the exact conflicting span in the previous text.\n"
            "If no exact conflicting span can be identified and the previous text already conveys the source fact faithfully, keep the wording unchanged.\n"
            "If a factual conflict is clear, correct it even when a sentence-level rewrite of the affected sentence or affected local region is needed.\n"
            "Preserve unaffected names, titles, years, identifiers, formulas, and parenthetical qualifiers unless they are themselves part of the factual error.\n"
            "Do not turn natural wording into awkward predicate-style expressions."
        ),
        "revision_requirements": [
            "Output the complete revised text, not just the edits.",
            "If no change is strictly necessary, output the previous text unchanged.",
            "Keep all correct source-supported information from the previous text.",
            "When a factual conflict is clear, prioritise correcting the fact over preserving local word order.",
            "Do not canonicalise titles, names, years, identifiers, formulas, or parenthetical qualifiers unless factual correctness requires it.",
            "Use only facts supported by the source triples.",
            "Do not mention the verifier, feedback, or triples in the final text.",
            'Output only one valid JSON object with exactly this schema: {"full-text": "..."}',
        ],
    },
}


def _append_guidance(text: str, extra_lines: list[str]) -> str:
    if not extra_lines:
        return text
    return text + "\n" + "\n".join(extra_lines)


def _dedupe_lines(lines: list[str]) -> list[str]:
    seen = set()
    kept: list[str] = []
    for line in lines:
        if line in seen:
            continue
        seen.add(line)
        kept.append(line)
    return kept


P2_SHARED_MISSING_GUIDANCE: list[str] = []

P2_SHARED_EXTRA_GUIDANCE: list[str] = []

P2_SHARED_INCORRECT_GUIDANCE: list[str] = []

P2_SHARED_REQUIREMENTS: list[str] = []


P2_SCOPE_PROFILES: Dict[str, Dict[str, Any]] = {
    "p2a": {
        "label": "P2A Original Scope",
        "intent": "Original P2-style scope control with local edits, affected-region rewriting, and wrapper recovery.",
        "task_lines": [
            "Prefer the smallest revision scope that fully fixes the issue.",
            "If the issue is local, edit only the affected span or clause. If the wording around an issue is broadly broken, recover only the affected region of text rather than rewriting unrelated content.",
        ],
        "missing_guidance": [
            "Integrate it into the affected local region when possible.",
            "Do not expand unrelated content just to patch one missing fact.",
        ],
        "extra_guidance": [
            "Avoid changing unaffected content outside the local span needed to fix the issue.",
            "Preserve nearby source-supported material unless a broader local rewrite is clearly necessary.",
        ],
        "incorrect_guidance": [
            "When a correction is needed, make the smallest local change that restores factual correctness and natural wording.",
            "Avoid changing unaffected content outside the local span needed to fix the issue.",
        ],
        "revision_requirements": [
            "If the previous text contains a broken wrapper such as ERROR or malformed JSON fragments, first recover the intended natural-language content from it, then revise the facts. Still return the final answer in the required JSON format.",
        ],
    },
    "p2b": {
        "label": "P2B Hardened Scope",
        "intent": "Hardened scope control: stay local when possible, and restrict broader rewrites to the affected region only.",
        "task_lines": [
            "Prefer the smallest revision scope that fully fixes the issue.",
            "If a local fix is enough, keep the revision local to the affected span or clause.",
            "If the wording around the issue is broadly broken, rewrite only the affected region and keep neighbouring source-supported content unchanged.",
        ],
        "missing_guidance": [
            "Add a listed fact only when it is not already faithfully expressed in the previous text.",
            "Integrate it into the affected region rather than expanding unrelated content.",
            "If a missing fact cannot be expressed cleanly with a local insertion, revise only the affected region needed to express it naturally.",
        ],
        "extra_guidance": [
            "Remove only the unsupported part needed to fix the issue.",
            "Keep unaffected surrounding content unchanged unless a broader local rewrite is clearly necessary.",
            "Preserve nearby source-supported material, even when you rewrite the affected region.",
        ],
        "incorrect_guidance": [
            "Locate the specific conflicting span in the previous text.",
            "If a local fix is enough, do not rewrite the full sentence.",
            "If the wording around the issue is broadly broken, rewrite only the affected region needed to express the fact correctly.",
            "Avoid changing unaffected surrounding content outside the affected region.",
        ],
        "revision_requirements": [
            "Do not rewrite a full sentence or neighbouring sentence unless the affected region itself is broadly broken.",
            "If the previous text contains a broken wrapper such as ERROR or malformed JSON fragments, first recover the intended natural-language content from it, then revise the facts. Still return the final answer in the required JSON format.",
        ],
        "overrides": {
            "p1b_validate_preserve": {
                "task_lines": [
                    "Produce a complete revised text that is a faithful lexicalisation of the source triples.",
                    "The source triples are the only facts allowed in the output.",
                    "Use the verifier feedback to inspect possible issues, then check each highlighted issue against the source triples and the previous text before deciding whether any revision is necessary.",
                    "The verifier feedback only identifies possible problem areas; it does not list all source-supported content that should remain in the final text.",
                    "Keep any wording that already faithfully expresses a fact supported by the source triples, unless it conflicts with another source-supported fact.",
                    "When revision is needed, prefer the smallest revision scope that fully fixes the issue: keep local fixes local, and rewrite only the affected region when the surrounding wording is broadly broken.",
                ],
                "missing_guidance": (
                    "For each listed missing fact, first check whether that fact is already faithfully expressed in the previous text.\n"
                    "If it is already expressed, keep the existing wording.\n"
                    "If it is genuinely missing, add it in the affected region without expanding unrelated content.\n"
                    "If a local insertion is not enough, revise only the affected region needed to express it naturally."
                ),
                "extra_guidance": (
                    "For each listed extra fact, verify that it is truly unsupported by the source triples.\n"
                    "If the wording differs but the meaning is source-supported, keep it.\n"
                    "Remove only the unsupported meaning that must change, and keep unaffected surrounding content unchanged unless a broader local rewrite is clearly necessary."
                ),
                "incorrect_guidance": (
                    "For each listed incorrect fact, compare the previous text with the source triples and determine whether the relevant claim is actually inconsistent.\n"
                    "If the previous text already expresses the source fact faithfully, keep it.\n"
                    "If a correction is needed, make the smallest local change that restores factual correctness and natural wording.\n"
                    "If the wording around the issue is broadly broken, rewrite only the affected region needed to express the fact correctly, without rewriting unrelated content."
                ),
                "revision_requirements": [
                    "Output the complete revised text, not just the edits.",
                    "If the previous text already faithfully expresses the source facts, output it unchanged.",
                    "Keep all correct source-supported information from the previous text.",
                    "Do not rewrite a full sentence or neighbouring sentence unless the affected region itself is broadly broken.",
                    "If the previous text contains a broken wrapper such as ERROR or malformed JSON fragments, first recover the intended natural-language content from it, then revise the facts. Still return the final answer in the required JSON format.",
                    "Use only facts supported by the source triples.",
                    "Do not mention the verifier, feedback, or triples in the final text.",
                    'Output only one valid JSON object with exactly this schema: {"full-text": "..."}',
                ],
            },
            "p1c_validate_preserve": {
                "task_lines": [
                    "Produce a complete revised text that is a faithful lexicalisation of the source triples.",
                    "The source triples are the only facts allowed in the output.",
                    "Use the verifier feedback as a signal to inspect the previous text, but decide whether to revise by checking the source triples and the previous text together.",
                    "The verifier feedback points to possible issues, not to all source-supported content that should remain in the text.",
                    "If the previous text already faithfully expresses a source-supported fact, keep that wording unchanged.",
                    "If revision is needed, prefer the smallest revision scope that fully fixes the issue: keep local fixes local, and rewrite only the affected region when the surrounding wording is broadly broken.",
                ],
                "missing_guidance": (
                    "If a listed missing fact is already faithfully expressed in the previous text, do not add it again.\n"
                    "If it is not expressed, add it in the affected region without expanding unrelated content.\n"
                    "If a local insertion is not enough, revise only the affected region needed to express it naturally."
                ),
                "extra_guidance": (
                    "If a listed extra fact is truly unsupported by the source triples, remove it.\n"
                    "If the wording is different but the meaning is source-supported, keep it.\n"
                    "Remove only the unsupported meaning that must change, and keep unaffected surrounding content unchanged unless a broader local rewrite is clearly necessary."
                ),
                "incorrect_guidance": (
                    "If the previous text already expresses the source fact faithfully, do not rewrite it only to mirror the predicate label more literally.\n"
                    "If a correction is needed, make the smallest local change that restores factual correctness and natural wording.\n"
                    "If the wording around the issue is broadly broken, rewrite only the affected region needed to express the fact correctly, without rewriting unrelated content."
                ),
                "revision_requirements": [
                    "Output the complete revised text, not just the edits.",
                    "If the previous text already faithfully expresses the source facts, output it unchanged.",
                    "Keep all correct source-supported information from the previous text.",
                    "Do not rewrite a full sentence or neighbouring sentence unless the affected region itself is broadly broken.",
                    "If the previous text contains a broken wrapper such as ERROR or malformed JSON fragments, first recover the intended natural-language content from it, then revise the facts. Still return the final answer in the required JSON format.",
                    "Use only facts supported by the source triples.",
                    "Do not mention the verifier, feedback, or triples in the final text.",
                    'Output only one valid JSON object with exactly this schema: {"full-text": "..."}',
                ],
            },
        },
    },
    "p2c": {
        "label": "P2C Tight Region + Merge",
        "intent": "Strong scope control with light merge/preserve discipline on top of the hardened affected-region policy.",
        "task_lines": [
            "Prefer the smallest revision scope that fully fixes the issue.",
            "If a local fix is enough, keep the revision local to the affected span or clause.",
            "If the wording around the issue is broadly broken, rewrite only the affected region and keep unaffected neighbouring content unchanged.",
            "Preserve source-supported wording when it does not need to change for factual correctness.",
        ],
        "missing_guidance": [
            "Add a listed fact only when it is not already faithfully expressed in the previous text.",
            "Integrate it into the affected region rather than expanding unrelated content.",
            "If a missing fact cannot be expressed cleanly with a local insertion, revise only the affected region needed to express it naturally.",
            "When adding a genuinely missing fact, merge it into an existing sentence or clause whenever possible instead of creating a detached patch sentence.",
        ],
        "extra_guidance": [
            "Remove only the unsupported part needed to fix the issue.",
            "Keep unaffected surrounding content unchanged unless a broader local rewrite is clearly necessary.",
            "Preserve nearby source-supported wording and exact strings when they do not need to change for factual correctness.",
        ],
        "incorrect_guidance": [
            "Locate the specific conflicting span in the previous text.",
            "If a local fix is enough, do not rewrite the full sentence.",
            "If the wording around the issue is broadly broken, rewrite only the affected region needed to express the fact correctly.",
            "Preserve source-supported wording around the local fix when it does not need to change for factual correctness.",
            "Prefer integrating the correction into the existing sentence or clause rather than adding a detached repair sentence.",
            "Avoid changing unaffected content outside the affected region.",
        ],
        "revision_requirements": [
            "Preserve source-supported wording and exact strings when they do not need to change for factual correctness.",
            "Avoid lazy additive patching when the correction can be merged into the existing wording.",
            "Do not rewrite a full sentence or neighbouring sentence unless the affected region itself is broadly broken.",
            "If the previous text contains a broken wrapper such as ERROR or malformed JSON fragments, first recover the intended natural-language content from it, then revise the facts. Still return the final answer in the required JSON format.",
        ],
        "overrides": {
            "p1b_validate_preserve": {
                "task_lines": [
                    "Produce a complete revised text that is a faithful lexicalisation of the source triples.",
                    "The source triples are the only facts allowed in the output.",
                    "Use the verifier feedback to inspect possible issues, then check each highlighted issue against the source triples and the previous text before deciding whether any revision is necessary.",
                    "The verifier feedback only identifies possible problem areas; it does not list all source-supported content that should remain in the final text.",
                    "Keep any wording that already faithfully expresses a fact supported by the source triples, unless it conflicts with another source-supported fact.",
                    "When revision is needed, prefer the smallest revision scope that fully fixes the issue: keep local fixes local, rewrite only the affected region when broader wording is broken, and preserve source-supported wording that does not need to change.",
                ],
                "missing_guidance": (
                    "For each listed missing fact, first check whether that fact is already faithfully expressed in the previous text.\n"
                    "If it is already expressed, keep the existing wording.\n"
                    "If it is genuinely missing, add it in the affected region without expanding unrelated content.\n"
                    "If a local insertion is not enough, revise only the affected region needed to express it naturally.\n"
                    "Merge the added fact into an existing sentence or clause whenever possible instead of creating a detached patch sentence."
                ),
                "extra_guidance": (
                    "For each listed extra fact, verify that it is truly unsupported by the source triples.\n"
                    "If the wording differs but the meaning is source-supported, keep it.\n"
                    "Remove only the unsupported meaning that must change, keep unaffected surrounding content unchanged unless a broader local rewrite is clearly necessary, and preserve source-supported wording and exact strings that do not need to change."
                ),
                "incorrect_guidance": (
                    "For each listed incorrect fact, compare the previous text with the source triples and determine whether the relevant claim is actually inconsistent.\n"
                    "If the previous text already expresses the source fact faithfully, keep it.\n"
                    "If a correction is needed, make the smallest local change that restores factual correctness and natural wording.\n"
                    "If the wording around the issue is broadly broken, rewrite only the affected region needed to express the fact correctly.\n"
                    "Preserve source-supported wording around the fix when it does not need to change, and prefer integrating the correction into the existing sentence or clause rather than adding a detached repair sentence."
                ),
                "revision_requirements": [
                    "Output the complete revised text, not just the edits.",
                    "If the previous text already faithfully expresses the source facts, output it unchanged.",
                    "Keep all correct source-supported information from the previous text.",
                    "Preserve source-supported wording and exact strings when they do not need to change for factual correctness.",
                    "Avoid lazy additive patching when the correction can be merged into the existing wording.",
                    "Do not rewrite a full sentence or neighbouring sentence unless the affected region itself is broadly broken.",
                    "If the previous text contains a broken wrapper such as ERROR or malformed JSON fragments, first recover the intended natural-language content from it, then revise the facts. Still return the final answer in the required JSON format.",
                    "Use only facts supported by the source triples.",
                    "Do not mention the verifier, feedback, or triples in the final text.",
                    'Output only one valid JSON object with exactly this schema: {"full-text": "..."}',
                ],
            },
            "p1c_validate_preserve": {
                "task_lines": [
                    "Produce a complete revised text that is a faithful lexicalisation of the source triples.",
                    "The source triples are the only facts allowed in the output.",
                    "Use the verifier feedback as a signal to inspect the previous text, but decide whether to revise by checking the source triples and the previous text together.",
                    "The verifier feedback points to possible issues, not to all source-supported content that should remain in the text.",
                    "If the previous text already faithfully expresses a source-supported fact, keep that wording unchanged.",
                    "If revision is needed, prefer the smallest revision scope that fully fixes the issue: keep local fixes local, rewrite only the affected region when broader wording is broken, and preserve source-supported wording that does not need to change.",
                ],
                "missing_guidance": (
                    "If a listed missing fact is already faithfully expressed in the previous text, do not add it again.\n"
                    "If it is not expressed, add it in the affected region without expanding unrelated content.\n"
                    "If a local insertion is not enough, revise only the affected region needed to express it naturally.\n"
                    "Merge the added fact into an existing sentence or clause whenever possible instead of creating a detached patch sentence."
                ),
                "extra_guidance": (
                    "If a listed extra fact is truly unsupported by the source triples, remove it.\n"
                    "If the wording is different but the meaning is source-supported, keep it.\n"
                    "Remove only the unsupported meaning that must change, keep unaffected surrounding content unchanged unless a broader local rewrite is clearly necessary, and preserve source-supported wording and exact strings that do not need to change."
                ),
                "incorrect_guidance": (
                    "If the previous text already expresses the source fact faithfully, do not rewrite it only to mirror the predicate label more literally.\n"
                    "If a correction is needed, make the smallest local change that restores factual correctness and natural wording.\n"
                    "If the wording around the issue is broadly broken, rewrite only the affected region needed to express the fact correctly.\n"
                    "Preserve source-supported wording around the fix when it does not need to change, and prefer integrating the correction into the existing sentence or clause rather than adding a detached repair sentence."
                ),
                "revision_requirements": [
                    "Output the complete revised text, not just the edits.",
                    "If the previous text already faithfully expresses the source facts, output it unchanged.",
                    "Keep all correct source-supported information from the previous text.",
                    "Preserve source-supported wording and exact strings when they do not need to change for factual correctness.",
                    "Avoid lazy additive patching when the correction can be merged into the existing wording.",
                    "Do not rewrite a full sentence or neighbouring sentence unless the affected region itself is broadly broken.",
                    "If the previous text contains a broken wrapper such as ERROR or malformed JSON fragments, first recover the intended natural-language content from it, then revise the facts. Still return the final answer in the required JSON format.",
                    "Use only facts supported by the source triples.",
                    "Do not mention the verifier, feedback, or triples in the final text.",
                    'Output only one valid JSON object with exactly this schema: {"full-text": "..."}',
                ],
            },
        },
    },
}


def _build_p1_p2_variant(
    base_key: str,
    profile_key: str,
    label: str,
    intent: str,
) -> Dict[str, Any]:
    base = PROMPT_VARIANTS[base_key]
    profile = P2_SCOPE_PROFILES[profile_key]
    override = profile.get("overrides", {}).get(base_key, {})
    return {
        "label": label,
        "intent": intent,
        "repair_system": base["repair_system"],
        "task_lines": override.get("task_lines", [*base["task_lines"], *profile["task_lines"]]),
        "missing_guidance": override.get(
            "missing_guidance",
            _append_guidance(
                base["missing_guidance"],
                [*P2_SHARED_MISSING_GUIDANCE, *profile["missing_guidance"]],
            ),
        ),
        "extra_guidance": override.get(
            "extra_guidance",
            _append_guidance(
                base["extra_guidance"],
                [*P2_SHARED_EXTRA_GUIDANCE, *profile["extra_guidance"]],
            ),
        ),
        "incorrect_guidance": override.get(
            "incorrect_guidance",
            _append_guidance(
                base["incorrect_guidance"],
                [*P2_SHARED_INCORRECT_GUIDANCE, *profile["incorrect_guidance"]],
            ),
        ),
        "revision_requirements": _dedupe_lines(
            override.get(
                "revision_requirements",
                [
                    *base["revision_requirements"],
                    *P2_SHARED_REQUIREMENTS,
                    *profile["revision_requirements"],
                ],
            )
        ),
    }


for _base_key, _short_name in (
    ("p1b_validate_preserve", "p1b"),
    ("p1c_validate_preserve", "p1c"),
):
    for _profile_key in ("p2a", "p2b", "p2c"):
        _profile = P2_SCOPE_PROFILES[_profile_key]
        _variant_key = f"{_short_name}_{_profile_key}"
        PROMPT_VARIANTS[_variant_key] = _build_p1_p2_variant(
            _base_key,
            _profile_key,
            f"{_short_name.upper()} + {_profile['label']}",
            f"{PROMPT_VARIANTS[_base_key]['intent']} {_profile['intent']}",
        )

VARIANT_ORDER = ["p1c_p2b"]


def get_prompt_variant(name: str) -> Dict[str, Any]:
    if name not in PROMPT_VARIANTS:
        known = ", ".join(VARIANT_ORDER)
        raise KeyError(f"Unknown prompt variant: {name}. Known variants: {known}")
    return PROMPT_VARIANTS[name]
