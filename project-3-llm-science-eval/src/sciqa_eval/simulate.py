"""Controlled response simulator used to validate the scorer.

The simulator is NOT a language model and its outputs are not claims about any
real model. It produces answers to the benchmark with *known* correctness: for
each response it records the intended label (``gold_label``) and error type, so
the automatic scorer can be measured against an answer key instead of being
trusted blindly. Three responder profiles with different error mixes make the
downstream reporting (per-responder accuracy, error taxonomy, feedback) realistic.

To grade real model outputs, skip this module and feed a JSONL file to
``evaluate_responses.py`` (see the README for the format).
"""

from __future__ import annotations

import math

import numpy as np

from .benchmark import Question
from .scorer import UNIT_INFO, normalize

PROFILES = {
    "SimResponder-A": {
        "numeric": {"correct": 0.78, "unit_missing": 0.04, "small_approx": 0.06, "power_of_ten": 0.02,
                    "unit_error": 0.02, "arithmetic": 0.03, "method_error": 0.04, "sign_error": 0.0,
                    "abstain": 0.01},
        "concept": {"correct": 0.65, "omit_one": 0.20, "omit_most": 0.05, "contradiction": 0.07, "abstain": 0.03},
    },
    "SimResponder-B": {
        "numeric": {"correct": 0.55, "unit_missing": 0.12, "small_approx": 0.08, "power_of_ten": 0.06,
                    "unit_error": 0.08, "arithmetic": 0.04, "method_error": 0.05, "sign_error": 0.0,
                    "abstain": 0.02},
        "concept": {"correct": 0.45, "omit_one": 0.25, "omit_most": 0.12, "contradiction": 0.13, "abstain": 0.05},
    },
    "SimResponder-C": {
        "numeric": {"correct": 0.35, "unit_missing": 0.08, "small_approx": 0.10, "power_of_ten": 0.10,
                    "unit_error": 0.04, "arithmetic": 0.12, "method_error": 0.15, "sign_error": 0.04,
                    "abstain": 0.02},
        "concept": {"correct": 0.25, "omit_one": 0.25, "omit_most": 0.20, "contradiction": 0.25, "abstain": 0.05},
    },
}

ABSTAIN_TEXT = "I am not able to determine this reliably from the information given."


def _fmt(value: float, rng, sig: int = 3, allow_times_ten: bool = True) -> str:
    if value != 0 and (abs(value) < 1e-3 or abs(value) >= 1e6):
        mantissa, exponent = f"{value:.{sig}e}".split("e")
        mantissa = f"{float(mantissa):g}"
        exponent = int(exponent)
        if allow_times_ten and rng.random() < 0.4:
            return f"{mantissa} × 10^{exponent}"
        return f"{mantissa}e{exponent}"
    return f"{value:.{sig}g}"


def _tol_eff(q: Question) -> float:
    return max(q.rel_tol, q.abs_tol / abs(q.answer))


def _numeric_value(q: Question, outcome: str, rng):
    """Return (value, unit_label, gold_label, gold_error) for one numeric outcome."""
    unit = q.display_unit
    ref = q.answer
    if outcome == "unit_error" and not q.unit:
        outcome = "arithmetic"
    if outcome == "small_approx":
        low, high = 1.5 * _tol_eff(q), 0.9 * q.partial_rel
        if q.partial_rel <= 0 or low >= high:
            outcome = "arithmetic"
    if outcome == "method_error" and not q.wrong_values:
        outcome = "arithmetic"

    if outcome == "correct":
        return ref, unit, "correct", None, 3
    if outcome == "unit_missing":
        if q.unit:
            return ref, "", "partial", "unit_missing", 3
        return ref, unit, "correct", None, 3
    if outcome == "small_approx":
        rel = rng.uniform(low, high) * rng.choice([-1, 1])
        return ref * (1 + rel), unit, "partial", "approximation_error", 4
    if outcome == "power_of_ten":
        k = int(rng.choice([-3, -2, -1, 1, 2, 3]))
        return ref * 10.0 ** k, unit, "incorrect", "power_of_ten_error", 4
    if outcome == "unit_error":
        dim = UNIT_INFO[normalize(q.unit)][0]
        wrong_unit = "kcal/mol" if dim in ("conc", "vol") else "nM"
        return ref, wrong_unit, "incorrect", "unit_error", 4
    if outcome == "arithmetic":
        factor = rng.uniform(1.3, 2.5) if rng.random() < 0.5 else rng.uniform(0.4, 0.75)
        return ref * factor, unit, "incorrect", "method_or_arithmetic_error", 4
    if outcome == "method_error":
        return float(rng.choice(q.wrong_values)), unit, "incorrect", "method_or_arithmetic_error", 4
    if outcome == "sign_error":
        return -ref, unit, "incorrect", "sign_error", 4
    raise ValueError(outcome)


def _numeric_response(q: Question, profile: dict, rng):
    outcomes = list(profile)
    outcome = rng.choice(outcomes, p=np.array(list(profile.values())) / sum(profile.values()))
    if outcome == "abstain":
        return ABSTAIN_TEXT, "incorrect", "abstention", outcome

    value, unit, label, etype, sig = _numeric_value(q, outcome, rng)
    # Correct answers are sometimes given in a different (equivalent) unit.
    if outcome == "correct" and q.unit and rng.random() < 0.25:
        dim, canon = UNIT_INFO[normalize(q.unit)]
        if dim == "conc":
            value, unit = value / 1e3, rng.choice(["uM", "µM"])
        elif dim == "vol":
            value, unit = value * 1e3, "nL"
    text_value = _fmt(value, rng, sig)
    answer = f"{text_value} {unit}".strip()

    r = rng.random()
    if r < 0.50:
        text = f"I'll use {q.method}. With {q.given}, this gives the result below.\n\nFinal answer: {answer}"
    elif r < 0.70:
        text = f"Given {q.given}, apply {q.method}.\nTherefore the answer is {answer}."
    elif r < 0.90:
        text = f"**Answer:** {answer}\n\nExplanation: {q.method}, using {q.given}."
    else:
        text = f"{answer} is what I get from {q.method} with {q.given}."
    return text, label, etype, outcome


def _concept_response(q: Question, profile: dict, rng):
    outcomes = list(profile)
    outcome = rng.choice(outcomes, p=np.array(list(profile.values())) / sum(profile.values()))
    n = len(q.groups)
    if outcome == "abstain":
        return ABSTAIN_TEXT, "incorrect", "abstention", outcome

    idx = list(range(n))
    paraphrase_idx = None
    if outcome == "correct":
        keep, label, etype = idx, "correct", None
        if rng.random() < 0.12:  # a valid group expressed without any rubric keyword
            paraphrase_idx = int(rng.integers(n))
    elif outcome == "omit_one":
        drop = int(rng.integers(n))
        keep, label, etype = [i for i in idx if i != drop], "partial", "missing_key_point"
    elif outcome == "omit_most":
        k = (n - 1) // 2
        keep = sorted(rng.choice(idx, size=k, replace=False).tolist()) if k else []
        label, etype = "incorrect", "missing_most_key_points"
    else:  # contradiction
        k = math.ceil(n / 2)
        keep = sorted(rng.choice(idx, size=k, replace=False).tolist())
        label, etype = "incorrect", "contradicts_accepted_science"

    sentences = []
    for i in keep:
        grp = q.groups[i]
        if i == paraphrase_idx:
            sentences.append(grp["paraphrase"])
        else:
            sentences.append(str(rng.choice(grp["sentences"])))
    if outcome == "contradiction":
        sentences.append(q.wrong_text)
    if outcome == "correct" and rng.random() < 0.15:
        sentences.append(q.negated_text)
    order = rng.permutation(len(sentences))
    body = " ".join(sentences[i] for i in order) if sentences else "This is a nuanced topic with several considerations."
    intro = str(rng.choice(["Short answer: ", "Here is a concise explanation. ", ""]))
    return intro + body, label, etype, outcome


def simulate_responses(questions: list[Question], n_per_question: int = 4, seed: int = 2026):
    """Return a list of response dicts (including the hidden gold label)."""
    rows = []
    for r_i, (name, profile) in enumerate(PROFILES.items()):
        for q_i, q in enumerate(questions):
            for rep in range(n_per_question):
                rng = np.random.default_rng([seed, r_i, q_i, rep])
                gen = _numeric_response if q.kind == "numeric" else _concept_response
                text, label, etype, outcome = gen(q, profile[q.kind], rng)
                rows.append({
                    "response_id": f"{name}-{q.qid}-{rep + 1}",
                    "responder": name,
                    "question_id": q.qid,
                    "text": text,
                    "gold_label": label,
                    "gold_error_type": etype,
                    "injected_outcome": outcome,
                })
    return rows
