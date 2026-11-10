"""End-to-end tests against the ground-truth injection log."""

import numpy as np
import pytest

from bioqc.clean import curate, naive_aggregate
from bioqc.evaluate import accuracy_vs_truth, detection_performance, overall_precision
from medchem_common.synthetic import generate_assay_records, generate_library


@pytest.fixture(scope="module")
def run():
    library = generate_library()
    raw, log = generate_assay_records(library, seed=7)
    flagged, compounds = curate(raw)
    return library, raw, log, flagged, compounds


def test_no_rows_lost_or_invented(run):
    _, raw, _, flagged, _ = run
    assert len(flagged) == len(raw)
    assert set(flagged["record_id"]) == set(raw["record_id"])


@pytest.mark.parametrize(
    "error_type",
    ["exact_duplicate", "invalid_smiles", "missing_value", "negative_value", "zero_value", "wrong_endpoint"],
)
def test_deterministic_problems_always_caught(run, error_type):
    _, _, log, flagged, _ = run
    det = detection_performance(flagged, log).set_index("error_type")
    assert det.loc[error_type, "recall"] == 1.0


def test_overall_recall_and_precision(run):
    _, _, log, flagged, _ = run
    stats = overall_precision(flagged, log)
    assert stats["overall_recall"] >= 0.90
    assert stats["precision"] >= 0.85


def test_curation_beats_naive_baseline(run):
    library, raw, _, _, compounds = run
    acc = accuracy_vs_truth(compounds, naive_aggregate(raw), library).set_index("pipeline")
    curated = acc.loc["curated (QC + median)"]
    naive = acc.loc["naive (units only, mean)"]
    assert curated["rmse"] < 0.7 * naive["rmse"]
    assert curated["share_error_gt_1"] < naive["share_error_gt_1"]


def test_curated_has_one_row_per_compound(run):
    _, _, _, _, compounds = run
    assert compounds["compound_id"].is_unique


def test_censored_only_compounds_have_no_potency(run):
    _, _, _, _, compounds = run
    cens_only = compounds[compounds["confidence"] == "censored_only"]
    assert cens_only["pic50_median"].isna().all()
    assert (cens_only["n_censored"] >= 1).all()


def test_pipeline_is_deterministic():
    library = generate_library()
    raw1, _ = generate_assay_records(library, seed=3)
    raw2, _ = generate_assay_records(library, seed=3)
    assert raw1.equals(raw2)
    f1, c1 = curate(raw1)
    f2, c2 = curate(raw2)
    assert np.allclose(c1["pic50_median"].fillna(-1), c2["pic50_median"].fillna(-1))
