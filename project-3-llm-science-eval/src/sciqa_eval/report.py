"""Figures and Markdown report for the evaluation."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import LABELS, category_table, cohen_kappa, confusion, responder_summary

BLUE, NAVY, ORANGE, GREY = "#2F6FC1", "#1F3A6E", "#E07B39", "#8A8F98"


def _style():
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "font.size": 10, "axes.titleweight": "bold"})


def plot_category_heatmap(scored: pd.DataFrame, path: Path) -> None:
    _style()
    tab = category_table(scored)
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(tab.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(tab.shape[1]))
    ax.set_xticklabels(tab.columns, rotation=20, ha="right")
    ax.set_yticks(range(tab.shape[0]))
    ax.set_yticklabels(tab.index)
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            v = tab.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="white" if v > 0.6 else "black", fontsize=9)
    ax.set_title("Mean score by topic and responder")
    fig.colorbar(im, ax=ax, label="mean score (0-1)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_error_taxonomy(scored: pd.DataFrame, path: Path) -> None:
    _style()
    bad = scored[scored["label"] != "correct"]
    tab = bad.pivot_table(index="responder", columns="error_type", values="response_id", aggfunc="count", fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 4.4))
    bottom = np.zeros(len(tab))
    palette = plt.cm.tab20(np.linspace(0, 1, max(len(tab.columns), 2)))
    for color, col in zip(palette, tab.columns):
        ax.bar(tab.index, tab[col], bottom=bottom, label=col, color=color)
        bottom += tab[col].to_numpy()
    ax.set_ylabel("responses not fully correct")
    ax.set_title("Error taxonomy by responder")
    ax.legend(fontsize=7.5, frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_confusion(cm: pd.DataFrame, kappa: float, path: Path) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(cm.to_numpy(), cmap="Blues")
    ax.set_xticks(range(len(cm.columns)))
    ax.set_xticklabels(cm.columns)
    ax.set_yticks(range(len(cm.index)))
    ax.set_yticklabels(cm.index)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm.iat[i, j]
            ax.text(j, i, int(v), ha="center", va="center", color="white" if v > cm.to_numpy().max() / 2 else "black")
    ax.set_xlabel("automatic scorer")
    ax.set_ylabel("answer key (gold)")
    ax.set_title(f"Scorer vs gold (kappa = {kappa:.2f})")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _md(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    return df.to_markdown(index=False, floatfmt=floatfmt)


def write_report(path: Path, *, questions, scored: pd.DataFrame, validated: pd.DataFrame | None, seed: int) -> None:
    summary = responder_summary(scored)
    n_num = sum(q.kind == "numeric" for q in questions)
    lines = [
        "# Scientific-answer evaluation report",
        "",
        f"Benchmark: **{len(questions)} questions** ({n_num} calculations with computed answer keys, "
        f"{len(questions) - n_num} concept questions with rubrics). "
        f"**{len(scored)} responses** from {scored['responder'].nunique()} simulated responders (seed {seed}).",
        "",
        "> The responders are controlled simulators with known error mixes, not real language models. "
        "They exist so the automatic scorer can be validated against an answer key. Real model outputs "
        "can be graded with `evaluate_responses.py`.",
        "",
        "## 1. Results by responder",
        "",
        _md(summary.rename(columns={"ci_low": "ci95_low", "ci_high": "ci95_high"})),
        "",
        "Mean score: correct = 1, partial = 0.5-0.8, incorrect = 0. CI is a bootstrap over responses.",
        "",
        "![category heatmap](category_heatmap.png)",
        "",
        "Each heatmap cell averages only 4-16 responses, so differences between individual cells are "
        "indicative, not statistically established; use the overall means and their intervals for comparisons.",
        "",
        "## 2. Error taxonomy",
        "",
        "![taxonomy](error_taxonomy.png)",
        "",
    ]
    tax = (scored[scored["label"] != "correct"]
           .pivot_table(index="error_type", columns="responder", values="response_id", aggfunc="count", fill_value=0)
           .reset_index())
    lines += [_md(tax, ".0f"), ""]

    if validated is not None:
        gold, pred = validated["gold_label"], validated["label"]
        cm = confusion(gold, pred)
        kappa, wkappa = cohen_kappa(gold, pred), cohen_kappa(gold, pred, weights="linear")
        agree = (gold == pred).mean()
        lines += [
            "## 3. Validating the automatic scorer against the answer key",
            "",
            f"- Exact agreement with gold labels: **{agree:.1%}** ({int((gold == pred).sum())} of {len(validated)})",
            f"- Cohen's kappa: **{kappa:.3f}**; linearly weighted kappa: **{wkappa:.3f}**",
            "",
            "![confusion](scorer_confusion.png)",
            "",
        ]
        by_kind = validated.groupby("kind").apply(
            lambda g: pd.Series({"responses": len(g), "agreement": (g["gold_label"] == g["label"]).mean()}),
            include_groups=False,
        ).reset_index()
        lines += [_md(by_kind), ""]
        bad = validated[validated["gold_label"] != validated["label"]]
        lines += ["### Where the scorer disagrees with the key", ""]
        if bad.empty:
            lines.append("No disagreements.")
        else:
            causes = (bad.groupby(["injected_outcome", "gold_label", "label"]).size()
                      .reset_index(name="count").sort_values("count", ascending=False))
            lines += [_md(causes.rename(columns={"label": "scorer_label", "injected_outcome": "simulated_behaviour"}), ".0f"), ""]
            lines += ["Examples:", ""]
            for r in bad.head(5).itertuples():
                snippet = r.text.replace("\n", " ")[:220]
                lines += [f"- `{r.response_id}` (gold **{r.gold_label}**, scorer **{r.label}**): {snippet}"]
            lines.append("")
        lines += [
            "### Reading the disagreements",
            "",
            "- *Paraphrase misses*: a correct concept answer worded without any rubric keyword is under-scored by the "
            "keyword rubric (gold correct, scorer partial). This is the main limitation of keyword rubrics.",
            "- *Extraction misses*: when the answer is not marked and is followed by restated inputs, last-number "
            "extraction can pick the wrong value. Requiring a `Final answer:` line removes this failure mode.",
            "",
            "Recommended workflow: use the automatic scorer for fast triage, and route concept answers scored "
            "`partial` to human (or model-assisted) review.",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")
