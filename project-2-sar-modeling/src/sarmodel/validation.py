"""Validation schemes and metrics.

Two questions are answered separately, because they have very different answers
in drug discovery:

* **Random CV** - "how well do we predict compounds drawn from the same
  chemical space we trained on?"  (interpolation; usually optimistic)
* **Leave-one-R2-group-out CV** - "how well do we predict a *new* amide
  substituent the model has never seen?"  (extrapolation to a new series; the
  scenario that matters when a project moves to new chemistry)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import LeaveOneGroupOut, RepeatedKFold


def regression_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    resid = y_pred - y_true
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    rho = spearmanr(y_true, y_pred).statistic if len(y_true) > 2 and np.ptp(y_pred) > 0 else np.nan
    return {
        "rmse": float(np.sqrt((resid ** 2).mean())),
        "mae": float(np.abs(resid).mean()),
        "r2": 1 - ss_res / ss_tot if ss_tot > 0 else np.nan,
        "spearman": float(rho) if rho == rho else np.nan,
    }


def bootstrap_ci(values, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap CI of the mean over folds (fold-level uncertainty)."""
    v = np.asarray([x for x in values if x == x], float)
    if len(v) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def random_splits(n: int, n_splits: int = 5, n_repeats: int = 5, seed: int = 0):
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    return list(rkf.split(np.arange(n)))


def group_splits(groups):
    return list(LeaveOneGroupOut().split(np.zeros(len(groups)), groups=groups))


def cross_validate(make_model, X: pd.DataFrame, y: pd.Series, splits, n_per_repeat: int | None = None):
    """Fit/predict over ``splits``.

    Returns per-fold metrics and out-of-fold predictions of the *first* pass over
    the data (first repeat), which is what the parity plots show.
    """
    fold_rows = []
    oof = pd.Series(np.nan, index=X.index)
    for k, (tr, te) in enumerate(splits):
        model = make_model()
        model.fit(X.iloc[tr], y.iloc[tr])
        pred = model.predict(X.iloc[te])
        m = regression_metrics(y.iloc[te], pred)
        m.update({"fold": k, "n_test": len(te)})
        fold_rows.append(m)
        if n_per_repeat is None or k < n_per_repeat:
            oof.iloc[te] = pred
    return pd.DataFrame(fold_rows), oof


def summarise(folds: pd.DataFrame) -> dict:
    out = {}
    for metric in ("rmse", "mae", "r2", "spearman"):
        lo, hi = bootstrap_ci(folds[metric].to_numpy())
        vals = folds[metric].to_numpy(dtype=float)
        out[metric] = float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")
        out[f"{metric}_lo"], out[f"{metric}_hi"] = lo, hi
    return out
