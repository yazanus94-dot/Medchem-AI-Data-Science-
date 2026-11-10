#!/usr/bin/env python3
"""Run the full bioactivity QC pipeline end to end.

Usage
-----
    python run_pipeline.py                      # generate synthetic data, run QC, write report
    python run_pipeline.py --input my_export.csv   # curate your own ChEMBL-style CSV (no evaluation)
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(HERE.parent / "common"))

import pandas as pd  # noqa: E402

from bioqc.clean import curate, naive_aggregate  # noqa: E402
from bioqc.evaluate import accuracy_vs_truth, detection_performance, overall_precision  # noqa: E402
from bioqc.report import plot_flag_counts, plot_parity, plot_recall, write_report  # noqa: E402
from medchem_common.synthetic import generate_assay_records, generate_library  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, help="CSV with ChEMBL-style columns (skips synthetic data and evaluation)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--outdir", type=Path, default=HERE / "results")
    args = parser.parse_args()

    data_dir = HERE / "data"
    args.outdir.mkdir(parents=True, exist_ok=True)

    if args.input:
        raw = pd.read_csv(args.input)
        flagged, compounds = curate(raw)
        flagged.to_csv(args.outdir / "record_level_flags.csv", index=False)
        compounds.to_csv(args.outdir / "curated_compounds.csv", index=False)
        print(f"Curated {len(compounds)} compounds from {len(raw)} records -> {args.outdir}")
        return

    data_dir.mkdir(exist_ok=True)
    library = generate_library()
    raw, log = generate_assay_records(library, seed=args.seed)
    library.to_csv(data_dir / "library_ground_truth.csv", index=False)
    raw.to_csv(data_dir / "raw_assay_records.csv", index=False)
    log.to_csv(data_dir / "injection_log_ANSWER_KEY.csv", index=False)

    flagged, compounds = curate(raw)
    naive = naive_aggregate(raw)
    detection = detection_performance(flagged, log)
    precision = overall_precision(flagged, log)
    accuracy = accuracy_vs_truth(compounds, naive, library)

    flagged.to_csv(args.outdir / "record_level_flags.csv", index=False)
    compounds.to_csv(args.outdir / "curated_compounds.csv", index=False)
    detection.to_csv(args.outdir / "detection_performance.csv", index=False)
    accuracy.to_csv(args.outdir / "accuracy_vs_truth.csv", index=False)

    plot_flag_counts(flagged, args.outdir / "flag_counts.png")
    plot_recall(detection, args.outdir / "detection_recall.png")
    plot_parity(compounds, naive, library, args.outdir / "potency_parity.png")
    write_report(
        args.outdir / "qc_report.md",
        flagged=flagged, compounds=compounds, detection=detection,
        precision=precision, accuracy=accuracy, seed=args.seed,
    )

    print(detection.to_string(index=False))
    print()
    print(accuracy.to_string(index=False))
    print(f"\nOverall recall {precision['overall_recall']:.1%}, precision {precision['precision']:.1%}")
    print(f"Report: {args.outdir / 'qc_report.md'}")


if __name__ == "__main__":
    main()
