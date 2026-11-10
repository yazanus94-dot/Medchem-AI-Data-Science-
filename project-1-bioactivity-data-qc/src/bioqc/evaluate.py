"""Evaluate the QC pipeline against the ground-truth injection log."""

from __future__ import annotations

import numpy as np
import pandas as pd


def detection_performance(flagged: pd.DataFrame, injection_log: pd.DataFrame) -> pd.DataFrame:
    """Recall per injected problem type, plus overall precision.

    A problem counts as *detected* when the corrupted record is excluded from
    the curated dataset. ``wrong_endpoint`` rows are legitimate measurements of
    a different quantity, so excluding them is the correct curation decision.
    """
    merged = flagged.merge(injection_log, on="record_id", how="left")
    rows = []
    for error_type, grp in merged[merged["error_type"].notna()].groupby("error_type"):
        rows.append(
            {
                "error_type": error_type,
                "injected": len(grp),
                "detected": int(grp["excluded"].sum()),
                "recall": grp["excluded"].mean(),
            }
        )
    table = pd.DataFrame(rows).sort_values("error_type").reset_index(drop=True)
    return table


def overall_precision(flagged: pd.DataFrame, injection_log: pd.DataFrame) -> dict:
    merged = flagged.merge(injection_log, on="record_id", how="left")
    excluded = merged[merged["excluded"]]
    injected = merged[merged["error_type"].notna()]
    # Censored records are legitimate, not excluded; "clean" = no injected problem.
    false_pos = excluded[excluded["error_type"].isna()]
    return {
        "records": len(merged),
        "injected_problems": len(injected),
        "excluded": len(excluded),
        "excluded_clean_records": len(false_pos),
        "precision": 1 - len(false_pos) / max(len(excluded), 1),
        "overall_recall": injected["excluded"].mean(),
    }


def accuracy_vs_truth(curated: pd.DataFrame, naive: pd.DataFrame, library: pd.DataFrame) -> pd.DataFrame:
    """RMSE / MAE of curated and naive compound potencies against true pIC50."""
    truth = library[["compound_id", "true_pic50"]]
    cur = curated[curated["n_replicates"] > 0].merge(truth, on="compound_id")
    nai = naive.merge(truth, on="compound_id")
    rows = []
    for name, d, col in [("curated (QC + median)", cur, "pic50_median"),
                         ("naive (units only, mean)", nai, "pic50_naive")]:
        err = d[col] - d["true_pic50"]
        rows.append(
            {
                "pipeline": name,
                "compounds": len(d),
                "rmse": float(np.sqrt((err ** 2).mean())),
                "mae": float(err.abs().mean()),
                "max_abs_error": float(err.abs().max()),
                "share_error_gt_1": float((err.abs() > 1.0).mean()),
            }
        )
    return pd.DataFrame(rows)
