"""Figures and the Markdown QC report."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .checks import ALL_FLAGS

PALETTE = {"blue": "#2F6FC1", "navy": "#1F3A6E", "orange": "#E07B39", "grey": "#8A8F98"}


def _style():
    plt.rcParams.update({
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "axes.titleweight": "bold",
    })


def plot_flag_counts(flagged: pd.DataFrame, path: Path) -> None:
    _style()
    counts = flagged[ALL_FLAGS].sum().sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh([c.replace("flag_", "") for c in counts.index], counts.values, color=PALETTE["blue"])
    for y, v in enumerate(counts.values):
        ax.text(v + 0.5, y, str(int(v)), va="center", fontsize=9)
    ax.set_xlabel("records flagged")
    ax.set_title("Quality flags raised on raw assay records")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_parity(curated: pd.DataFrame, naive: pd.DataFrame, library: pd.DataFrame, path: Path) -> None:
    _style()
    truth = library[["compound_id", "true_pic50"]]
    cur = curated[curated["n_replicates"] > 0].merge(truth, on="compound_id")
    nai = naive.merge(truth, on="compound_id")
    lo = min(truth["true_pic50"].min(), nai["pic50_naive"].min()) - 0.3
    hi = max(truth["true_pic50"].max(), nai["pic50_naive"].max()) + 0.3
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharex=True, sharey=True)
    for ax, d, col, title, color in [
        (axes[0], nai, "pic50_naive", "Naive (units only, mean)", PALETTE["orange"]),
        (axes[1], cur, "pic50_median", "Curated (QC + median)", PALETTE["blue"]),
    ]:
        rmse = np.sqrt(((d[col] - d["true_pic50"]) ** 2).mean())
        ax.scatter(d["true_pic50"], d[col], s=12, alpha=0.6, color=color)
        ax.plot([lo, hi], [lo, hi], color=PALETTE["grey"], lw=1)
        ax.set_title(f"{title}\nRMSE = {rmse:.2f}")
        ax.set_xlabel("true pIC50")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    axes[0].set_ylabel("estimated pIC50")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_recall(detection: pd.DataFrame, path: Path) -> None:
    _style()
    d = detection.sort_values("recall")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(d["error_type"], d["recall"], color=PALETTE["navy"])
    for y, (r, n) in enumerate(zip(d["recall"], d["injected"])):
        ax.text(min(r + 0.01, 1.0), y, f"{r:.0%} (n={n})", va="center", fontsize=9)
    ax.set_xlim(0, 1.25)
    ax.set_xlabel("share of injected problems excluded")
    ax.set_title("Detection rate by problem type")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_report(path: Path, *, flagged, compounds, detection, precision, accuracy, seed) -> None:
    flag_counts = flagged[ALL_FLAGS].sum()
    conf = compounds["confidence"].value_counts()
    lines = [
        "# Bioactivity data QC report",
        "",
        f"Synthetic ChEMBL-style IC50 export, random seed {seed}. "
        "Ground truth is known, so detection and accuracy are measured, not assumed.",
        "",
        "## 1. Dataset",
        "",
        f"- Raw records: **{len(flagged)}** for **{flagged['compound_id'].nunique()}** compounds",
        f"- Records excluded by QC: **{int(flagged['excluded'].sum())}**",
        f"- Records kept as censored (`>` relation, retained separately): **{int((~flagged['excluded'] & flagged['is_censored']).sum())}**",
        "",
        "## 2. Flags raised",
        "",
        "| Flag | Records |",
        "|---|---:|",
    ]
    lines += [f"| {k.replace('flag_', '')} | {int(v)} |" for k, v in flag_counts.items()]
    lines += [
        "",
        "![flags](flag_counts.png)",
        "",
        "## 3. Detection performance against the injection log",
        "",
        "| Problem type | Injected | Excluded | Recall |",
        "|---|---:|---:|---:|",
    ]
    lines += [
        f"| {r.error_type} | {r.injected} | {r.detected} | {r.recall:.1%} |"
        for r in detection.itertuples()
    ]
    lines += [
        "",
        f"Overall recall **{precision['overall_recall']:.1%}**, precision of exclusions "
        f"**{precision['precision']:.1%}** ({precision['excluded_clean_records']} clean records excluded "
        f"out of {precision['excluded']}).",
        "",
        "![recall](detection_recall.png)",
        "",
        "### Known limitations (by design of the data, not hidden)",
        "",
        "- **Single-replicate compounds** cannot be checked for outliers or mislabelled units; "
        "these are the main source of missed `unit_mislabel` and `outlier_replicate` records.",
        "- **Structure/ID conflicts** can only be detected when a compound has other records to compare with. "
        "When two structures are tied, both records are excluded (some clean records are lost).",
        "- **Two-replicate disagreements** cannot be attributed to either record, so a discordant pair excludes both. "
        "This, the tied-structure rule and the robust-outlier rule are why precision is below 100%: some "
        "genuine records are excluded to avoid keeping a wrong one. The thresholds are configurable trade-offs.",
        "- The SMILES check is specific to this chemical series. For arbitrary structures use RDKit sanitisation.",
        "",
        "## 4. Effect of curation on potency accuracy",
        "",
        "| Pipeline | Compounds | RMSE (pIC50) | MAE | Max abs error | Share with error > 1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    lines += [
        f"| {r.pipeline} | {r.compounds} | {r.rmse:.3f} | {r.mae:.3f} | {r.max_abs_error:.2f} | {r.share_error_gt_1:.1%} |"
        for r in accuracy.itertuples()
    ]
    lines += [
        "",
        "![parity](potency_parity.png)",
        "",
        "## 5. Curated dataset confidence",
        "",
        "| Confidence | Compounds |",
        "|---|---:|",
    ]
    lines += [f"| {k} | {v} |" for k, v in conf.items()]
    lines += [
        "",
        "`high` = at least 2 clean replicates with SD <= 0.5 log units; `medium` = 1 clean replicate or larger spread; "
        "`censored_only` = only `>` results (weaker than the assay ceiling).",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
