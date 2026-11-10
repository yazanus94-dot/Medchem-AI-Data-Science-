"""Beyond a single score: applicability domain, SAR recovery, cliffs, ranking."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .data import PHYSCHEM
from .validation import regression_metrics


def applicability_domain(measured: pd.DataFrame, splits, oof: pd.Series, n_bins: int = 3) -> pd.DataFrame:
    """Prediction error versus distance to the nearest *training* compound.

    Distance is Euclidean in standardised physicochemical space. Test compounds
    are binned into equal-count distance groups (near / mid / far).
    """
    scaled = pd.DataFrame(
        StandardScaler().fit_transform(measured[PHYSCHEM].astype(float)), index=measured.index
    ).to_numpy()
    rows = []
    for tr, te in splits:
        for i in te:
            d = np.sqrt(((scaled[tr] - scaled[i]) ** 2).sum(axis=1)).min()
            rows.append({"idx": i, "distance": d, "abs_error": abs(oof.iloc[i] - measured["pic50"].iloc[i])})
    df = pd.DataFrame(rows).drop_duplicates("idx")
    labels = ["near", "mid", "far"][:n_bins]
    df["bin"] = pd.qcut(df["distance"], n_bins, labels=labels)
    out = (
        df.groupby("bin", observed=True)
        .agg(n=("abs_error", "size"), mean_distance=("distance", "mean"),
             mae=("abs_error", "mean"), rmse=("abs_error", lambda e: float(np.sqrt((e ** 2).mean()))))
        .reset_index()
    )
    return out


def matched_pair_recovery(measured: pd.DataFrame, position: str, reference: str) -> pd.DataFrame:
    """Average substituent effect at ``position`` from matched molecular pairs.

    For every substituent X, find pairs of measured compounds that differ only
    at ``position`` (X vs ``reference``) and average the measured pIC50 change.
    Because the ground truth is known, the same pairs give the *true* average
    change, so recovery can be quantified.
    """
    others = [p for p in ("r1", "r2", "r3") if p != position]
    key = measured.set_index(others + [position])
    rows = []
    for sub in sorted(measured[position].unique()):
        if sub == reference:
            continue
        deltas_m, deltas_t = [], []
        for ctx, grp in measured[measured[position] == sub].groupby(others):
            ctx_key = ctx if isinstance(ctx, tuple) else (ctx,)
            try:
                ref_row = key.loc[ctx_key + (reference,)]
            except KeyError:
                continue
            for _, r in grp.iterrows():
                deltas_m.append(r["pic50"] - ref_row["pic50"])
                deltas_t.append(r["true_pic50"] - ref_row["true_pic50"])
        if deltas_m:
            rows.append({
                "position": position, "substituent": sub, "reference": reference,
                "n_pairs": len(deltas_m),
                "estimated_delta": float(np.mean(deltas_m)),
                "true_delta": float(np.mean(deltas_t)),
            })
    return pd.DataFrame(rows)


def activity_cliffs(
    measured: pd.DataFrame, threshold: float = 1.0, positions: tuple = ("r3",)
) -> pd.DataFrame:
    """Pairs that differ at exactly one position yet differ by >= ``threshold`` pIC50.

    By default only the small meta substituent (R3) is examined: swapping H for
    F/Cl/Me/OMe is a conservative change, so a >= 1 log-unit jump is
    surprising. (For R1/R2 many pairs differ that much simply because the
    substituents are chemically very different, which is not a cliff.)
    """
    rows = []
    recs = measured.to_dict("records")
    for a, b in itertools.combinations(recs, 2):
        diff = [p for p in ("r1", "r2", "r3") if a[p] != b[p]]
        if len(diff) != 1 or diff[0] not in positions:
            continue
        delta = b["pic50"] - a["pic50"]
        if abs(delta) >= threshold:
            rows.append({
                "changed_position": diff[0],
                "compound_a": a["compound_id"], "a_value": a[diff[0]],
                "compound_b": b["compound_id"], "b_value": b[diff[0]],
                "context": ", ".join(f"{p}={a[p]}" for p in ("r1", "r2", "r3") if p != diff[0]),
                "delta_pic50_measured": delta,
            })
    out = pd.DataFrame(rows)
    if len(out):
        out = out.reindex(out["delta_pic50_measured"].abs().sort_values(ascending=False).index)
    return out.reset_index(drop=True)


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values)."""
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out


def interaction_residuals(measured: pd.DataFrame, oof_additive: pd.Series, min_n: int = 4) -> pd.DataFrame:
    """Find substituent *combinations* an additive model systematically mispredicts.

    ``oof_additive`` are out-of-fold predictions of a model that can only add
    independent substituent effects. If a pair of substituents interacts (a
    pocket fit or a steric clash), compounds carrying that pair share a
    residual of the same sign. Groups are tested with a one-sample t-test and p-values are corrected for the
    many groups examined (Benjamini-Hochberg).
    """
    d = measured.assign(residual=measured["pic50"] - oof_additive)
    rows = []
    for a, b in (("r1", "r2"), ("r2", "r3"), ("r1", "r3")):
        for (va, vb), grp in d.groupby([a, b]):
            if len(grp) < min_n:
                continue
            sd = grp["residual"].std(ddof=1)
            t = grp["residual"].mean() / (sd / np.sqrt(len(grp))) if sd > 0 else np.nan
            rows.append({"pair": f"{a}={va} & {b}={vb}", "n": len(grp),
                         "mean_residual": grp["residual"].mean(), "t_stat": t})
    out = pd.DataFrame(rows).dropna(subset=["t_stat"])
    out["p_value"] = 2 * t_dist.sf(out["t_stat"].abs(), out["n"] - 1)
    out["q_value_bh"] = benjamini_hochberg(out["p_value"].to_numpy())
    return out.sort_values("p_value").reset_index(drop=True)


def prospective_ranking(model, X_train, y_train, X_virtual, virtual: pd.DataFrame, top_n: int = 10) -> dict:
    """Train on all measured compounds, rank the hidden virtual set, compare to truth."""
    model.fit(X_train, y_train)
    pred = model.predict(X_virtual)
    truth = virtual["true_pic50"].to_numpy()
    metrics = regression_metrics(truth, pred)
    top_pred = np.argsort(-pred)[:top_n]
    top_true = set(np.argsort(-truth)[:top_n])
    hits = len(top_true.intersection(top_pred))
    expected = top_n * top_n / len(virtual)
    return {
        **metrics,
        "top_n": top_n,
        "hits_in_true_top_n": hits,
        "expected_by_chance": expected,
        "enrichment_factor": hits / expected if expected > 0 else float("nan"),
        "predictions": pred,
    }


def feature_importance(make_model, X: pd.DataFrame, y: pd.Series, seed: int = 0, n_repeats: int = 10) -> pd.DataFrame:
    """Permutation importance (decrease in R^2) on a held-out 25% split."""
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=seed)
    model = make_model().fit(X_tr, y_tr)
    res = permutation_importance(model, X_te, y_te, n_repeats=n_repeats, random_state=seed, scoring="r2")
    return (
        pd.DataFrame({"feature": X.columns, "importance": res.importances_mean, "sd": res.importances_std})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
