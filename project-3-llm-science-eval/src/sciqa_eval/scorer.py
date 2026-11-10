"""Automatic, rubric-based scoring of free-text scientific answers.

Labels
------
``correct``    full credit (score 1.0)
``partial``    right idea but incomplete or imprecise (score 0.5-0.8)
``incorrect``  wrong, contradictory, missing or abstained (score 0.0)

Numeric answers are extracted from the text (preferring an explicit
"Final answer:"), converted to the expected unit and compared with the reference
under a per-question tolerance. Concept answers are scored by keyword-group
coverage plus detection of scientifically wrong claims, with a same-sentence
negation guard so that "it is a myth that X" is not penalised as "X".

Known limitation (measured in the report): a keyword rubric cannot recognise a
correct answer that is worded with none of the expected terms. Use it as a fast
first pass and route low-confidence items to human or model-assisted review.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .benchmark import Question

UNIT_INFO = {
    "pm": ("conc", 1e-3), "nm": ("conc", 1.0), "um": ("conc", 1e3), "mm": ("conc", 1e6), "m": ("conc", 1e9),
    "nl": ("vol", 1e-3), "ul": ("vol", 1.0), "ml": ("vol", 1e3),
    "kcal/mol": ("energy", 1.0), "kj/mol": ("energy", 0.239006),
    "g/mol": ("mass", 1.0), "da": ("mass", 1.0),
    "fold": ("ratio", 1.0), "%": ("percent", 1.0),
}

_NUM = r"(?<![a-z0-9.])[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?"
_UNITS = "|".join(sorted((re.escape(u) for u in UNIT_INFO), key=len, reverse=True))
QUANTITY = re.compile(rf"({_NUM})(?:\s*({_UNITS})(?![a-z/]))?")

NEGATION = re.compile(
    r"\b(not|never|no|isn't|incorrect|incorrectly|false|myth|misconception|wrong|cannot|doesn't|don't)\b"
)
ABSTAIN = re.compile(
    r"(i (do not|don't|am not able to|cannot|can't) (know|determine|answer|say)|"
    r"not enough information|insufficient information|unable to (determine|answer))"
)


def normalize(text: str) -> str:
    t = text.lower()
    t = t.replace("µ", "u").replace("μ", "u")
    t = t.replace("−", "-").replace("–", "-").replace("—", "-").replace("‑", "-")
    t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", t)
    t = re.sub(r"(\d(?:\.\d+)?)\s*[x×*]\s*10\s*\^?\s*\(?\s*([-+]?\d+)\s*\)?", r"\1e\2", t)
    return t


@dataclass
class Score:
    label: str
    score: float
    error_type: str | None
    extracted: str = ""
    explicit_final: bool = False


def extract_quantity(text: str):
    """Return ``(value, unit_token_or_None, explicit_flag)`` or ``None``."""
    norm = normalize(text)
    finals = list(
        re.finditer(r"(?:final answer|answer)(?:\*\*)?\s*[:=\-]\s*(?:\*\*)?\s*([^\n]*)", norm)
    )
    if finals:
        m = QUANTITY.search(finals[-1].group(1))
        if m:
            return float(m.group(1)), m.group(2), True
    # "the answer is X" / "therefore ... X" style statements.
    stated = list(re.finditer(r"(?:the answer is|therefore[^\n.]*?(?:is|=)|result is)\s*(?:\*\*)?\s*", norm))
    for m in reversed(stated):
        q = QUANTITY.match(norm, m.end())
        if q:
            return float(q.group(1)), q.group(2), False
    # A response that opens with the number ("7.6 is what I get from ...") leads with its
    # answer; the numbers that follow are restated inputs, so take the first quantity.
    lead = QUANTITY.match(norm.lstrip(" *\n"))
    if lead:
        return float(lead.group(1)), lead.group(2), False
    matches = list(QUANTITY.finditer(norm))
    if matches:
        m = matches[-1]
        return float(m.group(1)), m.group(2), False
    return None


def _tolerance(q: Question) -> float:
    return max(q.rel_tol * abs(q.answer), q.abs_tol)


def score_numeric(q: Question, text: str) -> Score:
    norm = normalize(text)
    found = extract_quantity(text)
    if found is None:
        if ABSTAIN.search(norm):
            return Score("incorrect", 0.0, "abstention")
        return Score("incorrect", 0.0, "no_answer")
    value, unit, explicit = found
    extracted = f"{value:g}" + (f" {unit}" if unit else "")
    unit_missing = False
    if q.unit:
        canon_dim, canon_factor = UNIT_INFO[normalize(q.unit)]
        if unit is None:
            unit_missing = True
        else:
            dim, factor = UNIT_INFO[unit]
            if dim != canon_dim:
                return Score("incorrect", 0.0, "unit_error", extracted, explicit)
            value = value * factor / canon_factor

    err = abs(value - q.answer)
    tol = _tolerance(q)
    if err <= tol:
        if unit_missing:
            return Score("partial", 0.8, "unit_missing", extracted, explicit)
        return Score("correct", 1.0, None, extracted, explicit)
    if q.partial_rel > 0 and err <= q.partial_rel * abs(q.answer):
        return Score("partial", 0.5, "approximation_error", extracted, explicit)
    if q.answer != 0 and abs(value + q.answer) <= tol:
        return Score("incorrect", 0.0, "sign_error", extracted, explicit)
    if q.answer != 0 and value > 0 and q.answer > 0:
        lg = math.log10(value / q.answer)
        if round(lg) != 0 and abs(lg - round(lg)) < 0.03:
            return Score("incorrect", 0.0, "power_of_ten_error", extracted, explicit)
    return Score("incorrect", 0.0, "method_or_arithmetic_error", extracted, explicit)


def _negated(norm: str, start: int) -> bool:
    """True if a negation word appears earlier in the same sentence as position ``start``."""
    boundaries = [m.end() for m in re.finditer(r"[.;!?]\s|\n", norm[:start])]
    sentence_start = boundaries[-1] if boundaries else 0
    return bool(NEGATION.search(norm[max(sentence_start, start - 80):start]))


def concept_hits(q: Question, norm: str) -> list[bool]:
    return [any(alt in norm for alt in grp["alts"]) for grp in q.groups]


def contradicts(q: Question, norm: str) -> bool:
    for pattern in q.forbidden:
        for m in re.finditer(pattern, norm):
            if not _negated(norm, m.start()):
                return True
    return False


def score_concept(q: Question, text: str) -> Score:
    norm = normalize(text)
    hits = concept_hits(q, norm)
    coverage = sum(hits) / len(hits)
    if contradicts(q, norm):
        return Score("incorrect", 0.0, "contradicts_accepted_science", f"coverage={coverage:.2f}")
    if coverage == 1.0:
        return Score("correct", 1.0, None, f"coverage={coverage:.2f}")
    if coverage >= 0.5:
        return Score("partial", round(coverage, 2), "missing_key_point", f"coverage={coverage:.2f}")
    if coverage == 0 and ABSTAIN.search(norm):
        return Score("incorrect", 0.0, "abstention", f"coverage={coverage:.2f}")
    return Score("incorrect", round(coverage, 2), "missing_most_key_points", f"coverage={coverage:.2f}")


def score_response(q: Question, text: str) -> Score:
    return score_numeric(q, text) if q.kind == "numeric" else score_concept(q, text)
