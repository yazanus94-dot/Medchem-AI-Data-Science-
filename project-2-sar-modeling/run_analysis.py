#!/usr/bin/env python3
"""Run the full SAR modelling analysis and write results/ (tables, figures, report)."""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(HERE.parent / "common"))

import pandas as pd  # noqa: E402

from sarmodel import analysis, report  # noqa: E402
from sarmodel.data import make_dataset  # noqa: E402
from sarmodel.features import build_features  # noqa: E402
from sarmodel.models import model_zoo  # noqa: E402
from sarmodel.validation import cross_validate, group_splits, random_splits, summarise  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--outdir", type=Path, default=HERE / "results")
    args = ap.parse_args()
    out = args.outdir
    out.mkdir(parents=True, exist_ok=True)

    library, measured, virtual = make_dataset(seed=args.seed)
    y = measured["pic50"]
    zoo = model_zoo(seed=args.seed)

    rand_splits = random_splits(len(measured), n_splits=5, n_repeats=5, seed=args.seed)
    grp_splits = group_splits(measured["r2"].to_numpy())

    # ---- 1. model comparison under two validation questions ---------------
    summary_rows, oofs = [], {}
    for name, (fset, factory) in zoo.items():
        X = build_features(measured, library, fset)
        for scheme, splits, per_rep in [("random CV", rand_splits, 5), ("leave-R2-group-out", grp_splits, None)]:
            folds, oof = cross_validate(factory, X, y, splits, n_per_repeat=per_rep)
            row = {"model": name, "scheme": scheme, "n_folds": len(folds), **summarise(folds)}
            summary_rows.append(row)
            oofs[(name, scheme)] = oof
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "model_comparison.csv", index=False)

    non_baseline = summary[(summary["model"] != "Mean baseline") & (summary["scheme"] == "leave-R2-group-out")]
    best = non_baseline.sort_values("rmse").iloc[0]["model"]
    fset_best, factory_best = zoo[best]
    X_best = build_features(measured, library, fset_best)

    report.plot_model_comparison(summary, out / "model_comparison.png")
    report.plot_parity(y, oofs[(best, "random CV")], oofs[(best, "leave-R2-group-out")], best, out / "parity_plots.png")
    report.plot_sar_heatmap(measured, out / "sar_heatmap.png")
    report.plot_residuals_by_group(measured, oofs[(best, "leave-R2-group-out")], out / "residuals_by_r2_group.png")

    # ---- 2. applicability domain -----------------------------------------
    ad = analysis.applicability_domain(measured, grp_splits, oofs[(best, "leave-R2-group-out")])
    ad.to_csv(out / "applicability_domain.csv", index=False)
    report.plot_applicability(ad, out / "applicability_domain.png")

    # ---- 3. SAR recovery from matched pairs ------------------------------
    mmp = pd.concat([
        analysis.matched_pair_recovery(measured, "r1", "Me"),
        analysis.matched_pair_recovery(measured, "r2", "Me"),
        analysis.matched_pair_recovery(measured, "r3", "H"),
    ], ignore_index=True)
    mmp.to_csv(out / "matched_pair_recovery.csv", index=False)
    report.plot_mmp_recovery(mmp, out / "matched_pair_recovery.png")

    # ---- 4. activity cliffs ----------------------------------------------
    cliffs = analysis.activity_cliffs(measured, threshold=1.0)
    n_pairs_r3 = sum(
        1 for _ in analysis.activity_cliffs(measured, threshold=0.0)["compound_a"]
    )
    # Ground-truth check: the planted steric clash is R2 = Pip together with R3 = Cl.
    if len(cliffs):
        cliffs["involves_planted_clash"] = cliffs["context"].str.contains("r2=Pip") & (
            (cliffs["a_value"] == "Cl") | (cliffs["b_value"] == "Cl")
        )
    cliffs.to_csv(out / "activity_cliffs.csv", index=False)

    # ---- 4b. interaction discovery from an additive model's residuals -------
    inter = analysis.interaction_residuals(measured, oofs[("Ridge (fragments)", "random CV")])
    inter.to_csv(out / "interaction_residuals.csv", index=False)

    # ---- 5. feature importance -------------------------------------------
    imp = analysis.feature_importance(factory_best, X_best, y, seed=args.seed)
    imp.to_csv(out / "feature_importance.csv", index=False)
    report.plot_importance(imp, out / "feature_importance.png")

    # ---- 6. prospective ranking on the hidden virtual set ------------------
    X_virtual = build_features(virtual, library, fset_best)
    pros = analysis.prospective_ranking(factory_best(), X_best, y, X_virtual, virtual, top_n=10)
    pred = pros.pop("predictions")
    pd.DataFrame([pros]).assign(model=best).to_csv(out / "prospective_ranking.csv", index=False)
    report.plot_prospective(
        virtual["true_pic50"], pred, pros["top_n"], out / "prospective_ranking.png",
        f"Prospective ranking ({best})\nhits in true top-{pros['top_n']}: {pros['hits_in_true_top_n']}",
    )

    # ---- console summary ----------------------------------------------------
    show = summary[["model", "scheme", "rmse", "rmse_lo", "rmse_hi", "r2", "spearman"]].round(3)
    print(show.to_string(index=False))
    print(f"\nBest model under leave-R2-group-out: {best}")
    print(f"Matched-pair recovery r = {mmp[['true_delta','estimated_delta']].corr().iloc[0,1]:.3f} over {len(mmp)} substituents")
    planted = int(cliffs["involves_planted_clash"].sum()) if len(cliffs) else 0
    print(f"R3 matched pairs: {n_pairs_r3}; cliff pairs (|delta| >= 1.0): {len(cliffs)}, of which {planted} involve the planted Pip/Cl clash")
    print("Top interaction candidates (additive-model residuals):")
    print(inter.head(6).round(2).to_string(index=False))
    print("Prospective ranking:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in pros.items()})
    print(ad.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
