#!/usr/bin/env python3
"""End-to-end demo: build the benchmark, simulate responders, score, validate the scorer, report."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

import pandas as pd  # noqa: E402

from sciqa_eval import report  # noqa: E402
from sciqa_eval.benchmark import build_benchmark  # noqa: E402
from sciqa_eval.feedback import build_feedback  # noqa: E402
from sciqa_eval.metrics import cohen_kappa, confusion, responder_summary  # noqa: E402
from sciqa_eval.scorer import score_response  # noqa: E402
from sciqa_eval.simulate import simulate_responses  # noqa: E402


def score_frame(rows, questions) -> pd.DataFrame:
    by_id = {q.qid: q for q in questions}
    out = []
    for r in rows:
        q = by_id[r["question_id"]]
        s = score_response(q, r["text"])
        out.append({
            "response_id": r["response_id"], "responder": r["responder"], "question_id": q.qid,
            "category": q.category, "kind": q.kind, "label": s.label, "score": s.score,
            "error_type": s.error_type, "extracted": s.extracted, "explicit_final": s.explicit_final,
        })
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--n-per-question", type=int, default=4)
    ap.add_argument("--outdir", type=Path, default=HERE / "results")
    args = ap.parse_args()
    out, data = args.outdir, HERE / "data"
    out.mkdir(parents=True, exist_ok=True)
    data.mkdir(exist_ok=True)

    questions = build_benchmark()
    (data / "benchmark.json").write_text(json.dumps([asdict(q) for q in questions], indent=2), encoding="utf-8")

    rows = simulate_responses(questions, n_per_question=args.n_per_question, seed=args.seed)
    # Responses file in the plug-in format (no gold labels inside).
    with open(data / "responses.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps({k: r[k] for k in ("response_id", "responder", "question_id", "text")}) + "\n")
    gold = pd.DataFrame(rows)[["response_id", "gold_label", "gold_error_type", "injected_outcome"]]
    gold.to_csv(data / "gold_labels_ANSWER_KEY.csv", index=False)

    scored = score_frame(rows, questions)
    validated = scored.merge(gold, on="response_id").merge(
        pd.DataFrame(rows)[["response_id", "text"]], on="response_id")
    scored.to_csv(out / "scored_responses.csv", index=False)
    summary = responder_summary(scored)
    summary.to_csv(out / "responder_summary.csv", index=False)

    cm = confusion(validated["gold_label"], validated["label"])
    kappa = cohen_kappa(validated["gold_label"], validated["label"])
    report.plot_category_heatmap(scored, out / "category_heatmap.png")
    report.plot_error_taxonomy(scored, out / "error_taxonomy.png")
    report.plot_confusion(cm, kappa, out / "scorer_confusion.png")
    report.write_report(out / "evaluation_report.md", questions=questions, scored=scored,
                        validated=validated, seed=args.seed)
    fb_dir = out / "feedback"
    fb_dir.mkdir(exist_ok=True)
    for name, text in build_feedback(scored).items():
        (fb_dir / f"{name}.md").write_text(text, encoding="utf-8")

    print(summary.round(3).to_string(index=False))
    agree = (validated["gold_label"] == validated["label"]).mean()
    print(f"\nScorer vs answer key: agreement {agree:.1%}, Cohen's kappa {kappa:.3f}")
    print(cm.to_string())


if __name__ == "__main__":
    main()
