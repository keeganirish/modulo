"""Conservative, programming-aware OCR ambiguity detection.

This module proposes alternatives for human review only. It never edits Node.text.
Python syntax is one weak ranking signal, not evidence that a candidate is intended.
"""
from __future__ import annotations
import ast
import re
from typing import Any

_CONFUSIONS = (
    (re.compile(r"(?<![=!<>])=(?!=)"), "==", "single '=' may be a comparison '=='"),
    (re.compile(r"=="), "=", "'==' may have been read as assignment '='"),
    (re.compile(r"%"), "0", "'%' may be confused with '0'"),
    (re.compile(r"(?<![A-Za-z0-9_])1(?![A-Za-z0-9_])"), "l", "digit '1' may be confused with lowercase 'l'"),
    (re.compile(r"(?<![A-Za-z0-9_])l(?![A-Za-z0-9_])"), "1", "lowercase 'l' may be confused with digit '1'"),
    (re.compile(r"\("), "[", "'(' may be confused with '['"),
    (re.compile(r"\)"), "]", "')' may be confused with ']'"),
    (re.compile(r"\["), "(", "'[' may be confused with '('"),
    (re.compile(r"\]"), ")", "']' may be confused with ')'"),
)


def _syntax_support(text: str, node_type: str) -> float:
    """Weak tie-breaker only; intentionally cannot determine intended meaning."""
    expression = text.strip()
    if node_type == "decision":
        expression = re.sub(r"^if\s+", "", expression, flags=re.I).rstrip(": ")
        mode = "eval"
    elif node_type == "iteration" and expression.upper().startswith("WHILE "):
        expression, mode = expression.split(None, 1)[1], "eval"
    elif node_type in ("process", "output"):
        if node_type == "output":
            expression = expression.split(None, 1)[1] if " " in expression else ""
            mode = "eval"
        else:
            mode = "exec"
    else:
        return 0.0
    try:
        ast.parse(expression, mode=mode)
        return 0.08
    except (SyntaxError, ValueError, RecursionError):
        return 0.0


def propose_ambiguities(text: str, node_type: str, existing: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Return deduplicated, ranked alternatives while retaining the original reading."""
    existing = existing or []
    proposals: dict[str, dict[str, Any]] = {}
    for candidate in existing:
        value = candidate.get("text") if isinstance(candidate, dict) else None
        if isinstance(value, str) and value.strip() and value.strip() != text.strip():
            value = value.strip()
            proposals[value] = {"text": value, "reason": "alternative OCR engine/preprocessing reading",
                                "score": float(candidate.get("score", 0.0)),
                                "confidence": float(candidate.get("confidence", 0.0)),
                                "syntax_support": _syntax_support(value, node_type)}
    for pattern, replacement, reason in _CONFUSIONS:
        match = pattern.search(text)
        if not match:
            continue
        alternative = text[:match.start()] + replacement + text[match.end():]
        if alternative.strip() == text.strip() or alternative in proposals:
            continue
        proposals[alternative] = {"text": alternative, "reason": reason, "score": 0.0,
                                  "confidence": 0.0, "syntax_support": _syntax_support(alternative, node_type)}
    # Syntax is deliberately a tiny tie-breaker and never auto-selects a reading.
    return sorted(proposals.values(), key=lambda item: (item["score"] + item["syntax_support"], item["confidence"]), reverse=True)[:8]


class ProgrammingOCRValidationService:
    name = "programming_ocr_validation"

    def run(self, context):
        graph = context.graph
        if graph is None:
            return context
        threshold = min(0.92, max(0.0, float(context.settings.OCR_MIN_CONFIDENCE)))
        for node in graph.nodes:
            # A user-reviewed reading is authoritative for this analysis. Keep the
            # original and suggestions attached for audit, but do not re-flag them.
            if "USER_CORRECTED" in node.flags:
                continue
            if node.type not in ("process", "input", "output", "decision", "iteration"):
                continue
            alternatives = propose_ambiguities(node.text, node.type, node.ocr_candidates)
            if not alternatives:
                continue
            # Existing candidate diversity is evidence of ambiguity. For synthesized
            # alternatives, only flag when OCR confidence is not comfortably high.
            has_engine_disagreement = any(item.get("reason") == "alternative OCR engine/preprocessing reading" for item in alternatives)
            has_character_confusion = any(item.get("reason") != "alternative OCR engine/preprocessing reading" for item in alternatives)
            suspicious = has_engine_disagreement or has_character_confusion or node.confidence < threshold
            if not suspicious:
                continue
            node.ocr_candidates = alternatives
            if "PROGRAMMING_OCR_AMBIGUITY" not in node.flags:
                node.flags.append("PROGRAMMING_OCR_AMBIGUITY")
            context.diagnose(
                    "warning", "PROGRAMMING_OCR_AMBIGUITY",
                    f"{node.id}: OCR may have confused programming symbols or characters; the original reading was preserved.",
                    "Review the original flowchart image and select the intended expression. Alternatives are suggestions, not automatic corrections.",
                    [node.id],
                )
        return context
