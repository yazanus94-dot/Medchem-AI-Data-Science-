#!/usr/bin/env python3
"""Grade your own responses.

Input: a JSONL file, one object per line:
    {"response_id": "m1-N01-1", "responder": "my-model", "question_id": "N01", "text": "...answer text..."}
Question IDs are listed in data/benchmark.json.

Usage:
    python evaluate_responses.py --responses my_responses.jsonl --outdir my_results
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

import pandas as pd  # noqa: E402

from run_evaluation import score_frame  # noqa: E402
from sciqa_eval import report  # noqa: E402
from sciqa_eval.benchmark import build_benchmark  # noqa: E402
from sciqa_eval.feedback import build_feedback  # noqa: E402
from sciqa_eval.metrics import responder_summary  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--responses", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, default=HERE / "my_results")
    args = ap.parse_args()

    rows = [json.loads(line) for line in args.responses.read_text(encoding="utf-8").splitlines() if line.strip()]
    questions = build_benchmark()
    known = {q.qid for q in questions}
    unknown = sorted({r["question_id"] for r in rows} - known)
    if unknown:
        sys.exit(f"Unknown question ids: {unknown}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    scored = score_frame(rows, questions)
    scored.to_csv(args.outdir / "scored_responses.csv", index=False)
    responder_summary(scored).to_csv(args.outdir / "responder_summary.csv", index=False)
    report.plot_category_heatmap(scored, args.outdir / "category_heatmap.png")
    report.plot_error_taxonomy(scored, args.outdir / "error_taxonomy.png")
    report.write_report(args.outdir / "evaluation_report.md", questions=questions, scored=scored,
                        validated=None, seed=0)
    fb_dir = args.outdir / "feedback"
    fb_dir.mkdir(exist_ok=True)
    for name, text in build_feedback(scored).items():
        (fb_dir / f"{name}.md").write_text(text, encoding="utf-8")
    print(responder_summary(scored).round(3).to_string(index=False))
    print(f"Report: {args.outdir / 'evaluation_report.md'}")


if __name__ == "__main__":
    main()
