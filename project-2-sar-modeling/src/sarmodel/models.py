"""Model zoo: one simple baseline, two linear models, two tree ensembles."""

from __future__ import annotations

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def model_zoo(seed: int = 0) -> dict:
    """Name -> (feature_set, estimator factory)."""
    return {
        "Mean baseline": ("physchem", lambda: DummyRegressor(strategy="mean")),
        "Ridge (physchem)": (
            "physchem",
            lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=[0.1, 1, 10, 100])),
        ),
        "Ridge (fragments)": (
            "fragments",
            lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=[0.1, 1, 10, 100])),
        ),
        "Random forest (both)": (
            "both",
            lambda: RandomForestRegressor(
                n_estimators=300, min_samples_leaf=2, random_state=seed, n_jobs=1
            ),
        ),
        "Gradient boosting (both)": (
            "both",
            lambda: GradientBoostingRegressor(
                n_estimators=250, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=seed
            ),
        ),
    }
