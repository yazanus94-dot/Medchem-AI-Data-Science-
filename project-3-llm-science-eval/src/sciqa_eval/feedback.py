"""Turn scored responses into actionable, per-responder improvement feedback."""

from __future__ import annotations

import pandas as pd

RECOMMENDATIONS = {
    "contradicts_accepted_science": (
        "Highest priority. The answer asserts something that contradicts accepted practice. Add counter-example "
        "training items for these topics, require the response to state caveats, and review whether the model "
        "is repeating a common misconception."),
    "unit_error": (
        "The number carries a unit of the wrong dimension. Add items that force explicit unit tracking "
        "(nM vs uM, uL vs mL, kcal/mol vs kJ/mol) and reward answers that state and convert units correctly."),
    "unit_missing": (
        "The value is right but the unit is missing. Require a unit with every reported quantity in the answer "
        "format and in the reward signal."),
    "power_of_ten_error": (
        "Factor-of-10^k errors point to unit-prefix slips. Add paired items that differ only by prefix and "
        "prompt a plausibility check (is this potency realistic for the assay range?)."),
    "approximation_error": (
        "Close but outside tolerance, usually premature rounding. Encourage carrying full precision through "
        "intermediate steps and rounding only the final answer."),
    "sign_error": (
        "Sign convention errors (for example binding free energy is negative for favourable binding). Add items "
        "that test sign interpretation, not just magnitude."),
    "method_or_arithmetic_error": (
        "Wrong formula or arithmetic. Compare against the listed common mistaken approaches for the affected "
        "questions and add worked-solution examples that show the correct formula and why the tempting "
        "alternative fails."),
    "missing_key_point": (
        "Mostly right but one required point is missing. Use the rubric group coverage to see which point is "
        "dropped and add examples where that point is essential."),
    "missing_most_key_points": (
        "The explanation is too thin to be useful. Reward complete, structured answers that address every "
        "rubric point, not only a single correct sentence."),
    "abstention": (
        "The responder declined to answer a question that is answerable from the information given. Check whether "
        "abstention is calibrated, or whether the model is being too cautious."),
    "no_answer": "No usable answer was found. Require an explicit 'Final answer:' line.",
}


def build_feedback(scored: pd.DataFrame, max_ids: int = 6) -> dict[str, str]:
    """Markdown feedback document per responder."""
    docs = {}
    for name, g in scored.groupby("responder"):
        bad = g[g["label"] != "correct"]
        lines = [
            f"# Feedback for {name}",
            "",
            f"{len(g)} responses graded: {int((g['label'] == 'correct').sum())} correct, "
            f"{int((g['label'] == 'partial').sum())} partial, {int((g['label'] == 'incorrect').sum())} incorrect "
            f"(mean score {g['score'].mean():.2f}).",
            "",
        ]
        if bad.empty:
            docs[name] = "\n".join(lines + ["No issues found."])
            continue
        counts = bad["error_type"].value_counts()
        lines += ["## Prioritised issues", ""]
        order = sorted(counts.index, key=lambda e: (e != "contradicts_accepted_science", -counts[e]))
        for rank, etype in enumerate(order, start=1):
            sub = bad[bad["error_type"] == etype]
            ids = ", ".join(sub["question_id"].value_counts().head(max_ids).index)
            lines += [
                f"### {rank}. `{etype}` ({len(sub)} responses, {len(sub) / len(g):.0%} of all)",
                f"Most affected questions: {ids}",
                "",
                RECOMMENDATIONS.get(etype, "Review these responses manually."),
                "",
            ]
        weakest = (
            g.groupby("category")["score"].mean().sort_values().head(2)
        )
        lines += ["## Weakest topic areas", ""]
        lines += [f"- {cat}: mean score {val:.2f}" for cat, val in weakest.items()]
        docs[name] = "\n".join(lines)
    return docs
