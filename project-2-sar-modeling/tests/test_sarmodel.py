import numpy as np
import pandas as pd
import pytest

from sarmodel.analysis import benjamini_hochberg, matched_pair_recovery
from sarmodel.data import make_dataset
from sarmodel.features import FEATURE_SETS, build_features
from sarmodel.models import model_zoo
from sarmodel.validation import (
    bootstrap_ci,
    cross_validate,
    group_splits,
    random_splits,
    regression_metrics,
)


@pytest.fixture(scope="module")
def data():
    return make_dataset(seed=11)


def test_dataset_partition_is_disjoint_and_complete(data):
    library, measured, virtual = data
    assert len(measured) == 300 and len(virtual) == 100
    assert set(measured["compound_id"]).isdisjoint(virtual["compound_id"])
    assert set(measured["compound_id"]) | set(virtual["compound_id"]) == set(library["compound_id"])
    assert "pic50" in measured and "pic50" not in virtual


def test_measurement_noise_is_roughly_as_specified(data):
    _, measured, _ = data
    noise = measured["pic50"] - measured["true_pic50"]
    assert abs(noise.mean()) < 0.06
    assert 0.25 < noise.std() < 0.35


@pytest.mark.parametrize("fset", FEATURE_SETS)
def test_feature_columns_identical_for_train_and_virtual(data, fset):
    library, measured, virtual = data
    a, b = build_features(measured, library, fset), build_features(virtual, library, fset)
    assert list(a.columns) == list(b.columns)
    assert not a.isna().any().any()


def test_one_hot_rows_have_one_active_group_per_position(data):
    library, measured, _ = data
    X = build_features(measured, library, "fragments")
    for pos in ("r1", "r2", "r3"):
        cols = [c for c in X.columns if c.startswith(pos + "_")]
        assert (X[cols].sum(axis=1) == 1).all()


def test_group_split_never_leaks_a_group(data):
    _, measured, _ = data
    groups = measured["r2"].to_numpy()
    for tr, te in group_splits(groups):
        assert set(groups[tr]).isdisjoint(groups[te])


def test_random_splits_cover_every_sample_once_per_repeat():
    splits = random_splits(50, n_splits=5, n_repeats=2, seed=0)
    assert len(splits) == 10
    for rep in range(2):
        seen = np.concatenate([te for _, te in splits[rep * 5:(rep + 1) * 5]])
        assert sorted(seen) == list(range(50))


def test_regression_metrics_known_values():
    m = regression_metrics([1, 2, 3, 4], [1, 2, 3, 4])
    assert m["rmse"] == 0 and m["mae"] == 0 and m["r2"] == 1 and m["spearman"] == pytest.approx(1)
    m = regression_metrics([1, 2, 3, 4], [2, 3, 4, 5])
    assert m["rmse"] == pytest.approx(1.0) and m["mae"] == pytest.approx(1.0)


def test_bootstrap_ci_brackets_the_mean():
    lo, hi = bootstrap_ci([0.4, 0.5, 0.45, 0.55, 0.5])
    assert lo < 0.48 < hi


def test_benjamini_hochberg_matches_hand_calculation():
    q = benjamini_hochberg(np.array([0.01, 0.04, 0.03, 0.20]))
    assert q == pytest.approx([0.04, 0.0533333, 0.0533333, 0.20], rel=1e-4)


def test_models_beat_mean_baseline_in_random_cv(data):
    library, measured, _ = data
    y = measured["pic50"]
    splits = random_splits(len(measured), 5, 1, seed=0)
    rmse = {}
    for name, (fset, factory) in model_zoo(seed=0).items():
        folds, _ = cross_validate(factory, build_features(measured, library, fset), y, splits)
        rmse[name] = folds["rmse"].mean()
    baseline = rmse.pop("Mean baseline")
    assert all(v < 0.7 * baseline for v in rmse.values())


def test_matched_pair_estimates_track_ground_truth(data):
    _, measured, _ = data
    mmp = matched_pair_recovery(measured, "r1", "Me")
    assert len(mmp) == 9
    assert np.corrcoef(mmp["estimated_delta"], mmp["true_delta"])[0, 1] > 0.95


def test_matched_pair_recovery_on_handmade_data():
    df = pd.DataFrame({
        "r1": ["Me", "F", "Me", "F"], "r2": ["Et", "Et", "Pip", "Pip"], "r3": ["H"] * 4,
        "pic50": [5.0, 5.6, 6.0, 6.4], "true_pic50": [5.0, 5.5, 6.0, 6.5],
    })
    out = matched_pair_recovery(df, "r1", "Me").iloc[0]
    assert out["n_pairs"] == 2
    assert out["estimated_delta"] == pytest.approx(0.5)
    assert out["true_delta"] == pytest.approx(0.5)
