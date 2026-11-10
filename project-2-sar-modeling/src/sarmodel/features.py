"""Feature construction. Columns are fixed from the full library so that train and
test matrices always have identical, aligned columns."""

from __future__ import annotations

import pandas as pd

from .data import PHYSCHEM

FEATURE_SETS = ("physchem", "fragments", "both")


def fragment_columns(library: pd.DataFrame) -> list[str]:
    cols = []
    for pos in ("r1", "r2", "r3"):
        cols += [f"{pos}_{v}" for v in sorted(library[pos].unique())]
    return cols


def build_features(df: pd.DataFrame, library: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    """Return a numeric feature matrix for ``df`` using the requested set."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"feature_set must be one of {FEATURE_SETS}")
    frag_cols = fragment_columns(library)
    parts = []
    if feature_set in ("physchem", "both"):
        parts.append(df[PHYSCHEM].astype(float))
    if feature_set in ("fragments", "both"):
        onehot = pd.DataFrame(
            {c: ((df[c.split("_", 1)[0]] == c.split("_", 1)[1]).astype(float)) for c in frag_cols},
            index=df.index,
        )
        parts.append(onehot)
    return pd.concat(parts, axis=1)
