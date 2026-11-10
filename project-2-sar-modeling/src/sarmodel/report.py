"""Figures for the SAR modelling project."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, NAVY, ORANGE, GREY = "#2F6FC1", "#1F3A6E", "#E07B39", "#8A8F98"


def _style():
    plt.rcParams.update({
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "axes.titleweight": "bold",
    })


def plot_model_comparison(summary: pd.DataFrame, path: Path) -> None:
    """RMSE for each model under random vs new-substituent validation (95% bootstrap CI)."""
    _style()
    models = list(summary["model"].unique())
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for off, scheme, color in [(-0.2, "random CV", BLUE), (0.2, "leave-R2-group-out", ORANGE)]:
        d = summary[summary["scheme"] == scheme].set_index("model").loc[models]
        err = np.vstack([d["rmse"] - d["rmse_lo"], d["rmse_hi"] - d["rmse"]])
        ax.bar(x + off, d["rmse"], width=0.38, yerr=err, capsize=3, label=scheme, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(" (", "\n(") for m in models], fontsize=8.5)
    ax.set_ylabel("RMSE (pIC50 units)")
    ax.set_title("Error depends on the validation question")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_parity(y, oof_random, oof_group, model_name: str, path: Path) -> None:
    _style()
    lo, hi = float(min(y.min(), oof_random.min(), oof_group.min())) - 0.2, float(max(y.max(), oof_random.max(), oof_group.max())) + 0.2
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharex=True, sharey=True)
    for ax, pred, title, color in [
        (axes[0], oof_random, "Random 5-fold CV", BLUE),
        (axes[1], oof_group, "Leave-one-R2-group-out", ORANGE),
    ]:
        rmse = np.sqrt(((pred - y) ** 2).mean())
        ax.scatter(y, pred, s=12, alpha=0.6, color=color)
        ax.plot([lo, hi], [lo, hi], color=GREY, lw=1)
        ax.set_title(f"{title}\nRMSE = {rmse:.2f}")
        ax.set_xlabel("measured pIC50")
    axes[0].set_ylabel(f"predicted pIC50 ({model_name})")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_sar_heatmap(measured: pd.DataFrame, path: Path) -> None:
    _style()
    pivot = measured.pivot_table(index="r1", columns="r2", values="pic50", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if v == v:
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", color="white", fontsize=8)
    ax.set_xlabel("R2 (amide N substituent)")
    ax.set_ylabel("R1 (para substituent)")
    ax.set_title("Measured SAR: mean pIC50 per R1 x R2")
    fig.colorbar(im, ax=ax, label="mean pIC50")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_residuals_by_group(measured: pd.DataFrame, oof_group: pd.Series, path: Path) -> None:
    _style()
    d = measured.assign(residual=oof_group - measured["pic50"])
    order = sorted(d["r2"].unique(), key=lambda g: d.loc[d["r2"] == g, "residual"].median())
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.boxplot([d.loc[d["r2"] == g, "residual"] for g in order], tick_labels=order, showfliers=False)
    ax.axhline(0, color=GREY, lw=1)
    ax.set_xlabel("R2 group held out")
    ax.set_ylabel("prediction - measured (pIC50)")
    ax.set_title("Where the model fails on a new R2 substituent")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_applicability(ad: pd.DataFrame, path: Path) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar(ad["bin"].astype(str), ad["rmse"], color=NAVY)
    for i, (r, n) in enumerate(zip(ad["rmse"], ad["n"])):
        ax.text(i, r + 0.005, f"{r:.2f}\n(n={n})", ha="center", fontsize=9)
    ax.set_xlabel("distance to nearest training compound (physchem space)")
    ax.set_ylabel("RMSE")
    ax.set_title("Applicability domain check")
    ax.set_ylim(0, ad["rmse"].max() * 1.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_mmp_recovery(mmp: pd.DataFrame, path: Path) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(5.5, 5))
    colors = {"r1": BLUE, "r2": ORANGE, "r3": NAVY}
    for pos, grp in mmp.groupby("position"):
        ax.scatter(grp["true_delta"], grp["estimated_delta"], s=30, color=colors[pos], label=pos.upper())
    lo = min(mmp["true_delta"].min(), mmp["estimated_delta"].min()) - 0.1
    hi = max(mmp["true_delta"].max(), mmp["estimated_delta"].max()) + 0.1
    ax.plot([lo, hi], [lo, hi], color=GREY, lw=1)
    r = np.corrcoef(mmp["true_delta"], mmp["estimated_delta"])[0, 1]
    ax.set_title(f"Matched-pair SAR recovery (r = {r:.2f})")
    ax.set_xlabel("true average change in pIC50")
    ax.set_ylabel("estimated from measured pairs")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_importance(imp: pd.DataFrame, path: Path, top: int = 12) -> None:
    _style()
    d = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.barh(d["feature"], d["importance"], xerr=d["sd"], color=BLUE, capsize=2)
    ax.set_xlabel("permutation importance (drop in R$^2$)")
    ax.set_title("Which features does the model rely on?")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_prospective(truth, pred, top_n: int, path: Path, title: str) -> None:
    _style()
    truth, pred = np.asarray(truth), np.asarray(pred)
    top_pred = set(np.argsort(-pred)[:top_n])
    top_true = set(np.argsort(-truth)[:top_n])
    fig, ax = plt.subplots(figsize=(5.5, 5))
    colors = [ORANGE if i in top_pred else GREY for i in range(len(truth))]
    ax.scatter(truth, pred, s=[40 if i in top_true else 14 for i in range(len(truth))], c=colors, alpha=0.75)
    lo, hi = min(truth.min(), pred.min()) - 0.2, max(truth.max(), pred.max()) + 0.2
    ax.plot([lo, hi], [lo, hi], color=GREY, lw=1)
    ax.set_xlabel("true pIC50 (hidden from the model)")
    ax.set_ylabel("predicted pIC50")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
