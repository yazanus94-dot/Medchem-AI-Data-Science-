import math

import pytest

from sciqa_eval import reference as ref
from sciqa_eval.benchmark import benchmark_by_id, build_benchmark
from sciqa_eval.metrics import cohen_kappa
from sciqa_eval.scorer import extract_quantity, score_response
from sciqa_eval.simulate import simulate_responses

Q = benchmark_by_id()


# ---------------------------------------------------------------- reference answers
def test_reference_values_by_hand():
    assert ref.pic50_from_ic50_nm(25) == pytest.approx(7.60206, abs=1e-4)
    assert ref.ic50_nm_from_pic50(6.3) == pytest.approx(501.19, abs=0.01)
    assert ref.cheng_prusoff_ki(120, 50, 10) == pytest.approx(20.0)
    assert ref.z_prime(100, 5, 10, 4) == pytest.approx(0.7)
    assert ref.ligand_efficiency(7.0, 28) == pytest.approx(0.34250)
    assert ref.lipophilic_ligand_efficiency(7.5, 3.1) == pytest.approx(4.4)
    assert ref.log2_fold_change(80, 20) == pytest.approx(2.0)
    assert ref.bonferroni_threshold(0.05, 20000) == pytest.approx(2.5e-6)
    assert ref.fractional_occupancy(50, 10) == pytest.approx(5 / 6)
    assert ref.stock_volume(10_000, 50, 100) == pytest.approx(0.5)
    assert ref.hill_fractional_inhibition(3.0) == pytest.approx(0.75)
    assert ref.selectivity_fold(4500, 15) == pytest.approx(300.0)
    assert ref.mean_pic50_from_ic50_nm([10, 100, 1000]) == pytest.approx(7.0)
    assert ref.lipinski_violations(540, 5.8, 2, 9) == 2


def test_delta_g_is_negative_and_about_minus_ten_point_nine():
    assert ref.delta_g_binding_kcal(10e-9) == pytest.approx(-10.9, abs=0.05)


def test_benjamini_hochberg_known_example():
    adj = ref.benjamini_hochberg([0.001, 0.008, 0.039, 0.041, 0.042, 0.060])
    assert adj == pytest.approx([0.006, 0.024, 0.0504, 0.0504, 0.0504, 0.060], abs=1e-6)
    assert all(a >= p for a, p in zip(adj, [0.001, 0.008, 0.039, 0.041, 0.042, 0.060]))


def test_hypergeometric_enrichment_is_small_and_valid():
    p = ref.hypergeometric_enrichment_p(20000, 500, 300, 25)
    assert 0 < p < 1e-3


def test_benchmark_ids_unique_and_wrong_values_really_wrong():
    qs = build_benchmark()
    assert len({q.qid for q in qs}) == len(qs) == 24
    for q in qs:
        if q.kind != "numeric":
            continue
        tol = max(q.rel_tol * abs(q.answer), q.abs_tol)
        for w in q.wrong_values:
            assert abs(w - q.answer) > tol, f"{q.qid}: listed wrong value is within tolerance"


# ---------------------------------------------------------------- extraction
@pytest.mark.parametrize("text,value,unit", [
    ("Final answer: 7.6", 7.6, None),
    ("Work shown... Final answer: 501 nM", 501.0, "nm"),
    ("**Answer:** 0.343 kcal/mol per heavy atom", 0.343, "kcal/mol"),
    ("7.6 is what I get from pIC50 = 9 - log10(IC50 in nM) with IC50 = 25 nM.", 7.6, None),
    ("Therefore the answer is 20 nM.", 20.0, "nm"),
    ("Final answer: 2.5 × 10^-6", 2.5e-6, None),
    ("Final answer: 1,200 nM", 1200.0, "nm"),
    ("Final answer: 0.5 µM", 0.5, "um"),
])
def test_extract_quantity(text, value, unit):
    v, u, _ = extract_quantity(text)
    assert v == pytest.approx(value)
    assert u == unit


def test_extract_quantity_none_when_no_number():
    assert extract_quantity("I am not sure.") is None


# ---------------------------------------------------------------- numeric scoring
def test_numeric_correct_and_unit_conversion():
    assert score_response(Q["N02"], "Final answer: 501 nM").label == "correct"
    assert score_response(Q["N02"], "Final answer: 0.501 uM").label == "correct"
    assert score_response(Q["N13"], "Final answer: 500 nL").label == "correct"


def test_numeric_unit_missing_is_partial_and_wrong_dimension_is_incorrect():
    s = score_response(Q["N02"], "Final answer: 501")
    assert (s.label, s.error_type) == ("partial", "unit_missing")
    s = score_response(Q["N02"], "Final answer: 501 kcal/mol")
    assert (s.label, s.error_type) == ("incorrect", "unit_error")


def test_numeric_error_taxonomy():
    assert score_response(Q["N03"], "Final answer: 20000 nM").error_type == "power_of_ten_error"
    assert score_response(Q["N11"], "Final answer: 10.9 kcal/mol").error_type == "sign_error"
    assert score_response(Q["N03"], "Final answer: 100 nM").error_type == "method_or_arithmetic_error"
    s = score_response(Q["N04"], "Final answer: 0.66")
    assert (s.label, s.error_type) == ("partial", "approximation_error")


def test_abstention_and_no_answer():
    assert score_response(Q["N01"], "I cannot determine this.").error_type == "abstention"
    assert score_response(Q["N01"], "It depends on the assay.").error_type == "no_answer"


def test_tolerance_edges():
    # Tolerance is max(2% relative, absolute): for N01 (answer 7.602) that is +/-0.152.
    assert score_response(Q["N01"], "Final answer: 7.62").label == "correct"
    assert score_response(Q["N01"], "Final answer: 7.74").label == "correct"
    assert score_response(Q["N01"], "Final answer: 7.80").label == "partial"
    assert score_response(Q["N01"], "Final answer: 8.5").label == "incorrect"


# ---------------------------------------------------------------- concept scoring
def _full_answer(q, idx=None):
    return " ".join(g["sentences"][0] for i, g in enumerate(q.groups) if idx is None or i in idx)


def test_concept_full_partial_wrong():
    q = Q["C01"]
    assert score_response(q, _full_answer(q)).label == "correct"
    assert score_response(q, _full_answer(q, {0, 1, 2})).label == "partial"
    s = score_response(q, _full_answer(q) + " " + q.wrong_text)
    assert (s.label, s.error_type) == ("incorrect", "contradicts_accepted_science")


def test_negation_guard_does_not_penalise_debunking_the_myth():
    q = Q["C03"]
    text = _full_answer(q) + " " + q.negated_text
    assert score_response(q, text).label == "correct"


def test_concept_empty_and_abstain():
    assert score_response(Q["C06"], "").label == "incorrect"
    assert score_response(Q["C06"], "I do not know.").error_type == "abstention"


# ---------------------------------------------------------------- scorer vs simulator key
def test_scorer_agrees_with_answer_key():
    rows = simulate_responses(build_benchmark(), n_per_question=4, seed=2026)
    gold = [r["gold_label"] for r in rows]
    pred = [score_response(Q[r["question_id"]], r["text"]).label for r in rows]
    agreement = sum(g == p for g, p in zip(gold, pred)) / len(rows)
    assert agreement >= 0.97
    assert cohen_kappa(gold, pred) >= 0.95
    # No response may be called correct when the key says incorrect or partial.
    false_credit = sum(p == "correct" and g != "correct" for g, p in zip(gold, pred))
    assert false_credit == 0


def test_simulation_is_deterministic_and_complete():
    a = simulate_responses(build_benchmark(), n_per_question=2, seed=5)
    b = simulate_responses(build_benchmark(), n_per_question=2, seed=5)
    assert [r["text"] for r in a] == [r["text"] for r in b]
    assert len(a) == 3 * 24 * 2


def test_cohen_kappa_extremes():
    assert cohen_kappa(["correct", "partial", "incorrect"], ["correct", "partial", "incorrect"]) == pytest.approx(1.0)
    assert math.isclose(cohen_kappa(["correct"] * 5 + ["incorrect"] * 5, ["correct", "incorrect"] * 5), 0.0, abs_tol=0.25)
