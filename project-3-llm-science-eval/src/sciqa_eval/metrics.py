"""Agreement and summary statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd

LABELS = ["incorrect", "partial", "correct"]  # ordinal


def confusion(gold, pred, labels=LABELS) -> pd.DataFrame:
    return pd.crosstab(
        pd.Categorical(gold, categories=labels), pd.Categorical(pred, categories=labels),
        rownames=["gold"], colnames=["scorer"], dropna=False,
    )


def cohen_kappa(gold, pred, labels=LABELS, weights: str | None = None) -> float:
    """Cohen's kappa; ``weights='linear'`` weights disagreements by ordinal distance."""
    cm = confusion(gold, pred, labels).to_numpy(dtype=float)
    n = cm.sum()
    k = len(labels)
    if weights == "linear":
        w = np.array([[abs(i - j) / (k - 1) for j in range(k)] for i in range(k)])
    else:
        w = 1.0 - np.eye(k)
    expected = np.outer(cm.sum(axis=1), cm.sum(axis=0)) / n
    denom = (w * expected).sum()
    return float(1.0 - (w * cm).sum() / denom) if denom > 0 else 1.0


def accuracy(gold, pred) -> float:
    return float((np.asarray(gold) == np.asarray(pred)).mean())


def bootstrap_ci_mean(values, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    v = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def responder_summary(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, g in scored.groupby("responder"):
        lo, hi = bootstrap_ci_mean(g["score"])
        rows.append({
            "responder": name,
            "responses": len(g),
            "mean_score": g["score"].mean(),
            "ci_low": lo, "ci_high": hi,
            "strict_accuracy": (g["label"] == "correct").mean(),
            "partial_share": (g["label"] == "partial").mean(),
            "incorrect_share": (g["label"] == "incorrect").mean(),
        })
    return pd.DataFrame(rows).sort_values("mean_score", ascending=False).reset_index(drop=True)


def category_table(scored: pd.DataFrame) -> pd.DataFrame:
    return scored.pivot_table(index="category", columns="responder", values="score", aggfunc="mean")
