"""Curation: decide which records are usable and aggregate to one value per compound."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .checks import ALL_FLAGS, REPLICATE_FLAGS, RECORD_LEVEL_FLAGS, check_replicates, run_record_checks

# Every flag blocks a record from the curated potency set.
BLOCKING_FLAGS = ALL_FLAGS


def curate(raw: pd.DataFrame):
    """Run all checks and aggregate.

    Returns
    -------
    flagged : DataFrame
        All input records with ``flag_*`` columns, ``pic50`` and ``excluded``
        (True if any blocking flag is set).
    compounds : DataFrame
        One row per compound with the curated potency summary.
    """
    flagged = run_record_checks(raw)
    flagged = check_replicates(flagged)
    flagged["excluded"] = flagged[BLOCKING_FLAGS].any(axis=1)
    flagged["flag_count"] = flagged[BLOCKING_FLAGS].sum(axis=1)

    usable = flagged[~flagged["excluded"] & ~flagged["is_censored"] & flagged["pic50"].notna()]
    agg = (
        usable.groupby("compound_id")["pic50"]
        .agg(n_replicates="count", pic50_median="median", pic50_mean="mean", pic50_sd="std")
        .reset_index()
    )
    agg["pic50_sd"] = agg["pic50_sd"].fillna(0.0)

    # Censored records are kept separately: they say "weaker than the ceiling".
    cens = flagged[~flagged["excluded"] & flagged["is_censored"]]
    cens_n = cens.groupby("compound_id").size().rename("n_censored").reset_index()
    compounds = agg.merge(cens_n, on="compound_id", how="outer")
    compounds["n_replicates"] = compounds["n_replicates"].fillna(0).astype(int)
    compounds["n_censored"] = compounds["n_censored"].fillna(0).astype(int)

    structures = (
        flagged[~flagged["flag_invalid_smiles"] & ~flagged["flag_structure_conflict"]]
        .groupby("compound_id")["smiles"]
        .agg(lambda s: s.value_counts().index[0])
        .rename("smiles")
        .reset_index()
    )
    compounds = compounds.merge(structures, on="compound_id", how="left")

    def confidence(row) -> str:
        if row["n_replicates"] >= 2 and row["pic50_sd"] <= 0.5:
            return "high"
        if row["n_replicates"] >= 1:
            return "medium"
        if row["n_censored"] >= 1:
            return "censored_only"
        return "none"

    compounds["confidence"] = compounds.apply(confidence, axis=1)
    return flagged, compounds.sort_values("compound_id").reset_index(drop=True)


def naive_aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    """Baseline for comparison: unit-aware, but NO quality control.

    Converts units, drops rows that cannot yield a number (missing/zero/negative
    or censored), and averages pIC50 per compound. Duplicates, wrong endpoints,
    outliers and mislabelled units are all kept.
    """
    df = run_record_checks(raw)
    ok = (
        ~df["flag_missing_value"]
        & ~df["flag_nonpositive_value"]
        & ~df["flag_unknown_unit"]
        & ~df["is_censored"]
    )
    return (
        df[ok].groupby("compound_id")["pic50"].mean().rename("pic50_naive").reset_index()
    )
