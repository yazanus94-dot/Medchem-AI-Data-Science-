"""Synthetic bioactivity data with a known ground-truth SAR.

Why synthetic?  With real public data you can never be sure what the "true"
activity of a compound is, so you cannot *prove* that a QC pipeline or a model
recovers it.  Here the true pIC50 of every compound is defined by an explicit,
documented rule (below), so every claim made by the projects can be checked
against ground truth.  The schema deliberately mirrors ChEMBL activity exports
(``standard_type``, ``standard_relation``, ``standard_value``,
``standard_units``), so the same code can be pointed at a real export.

Ground-truth SAR for the hypothetical target "TGT-1"
----------------------------------------------------
    pIC50 = 5.2 + a1[R1] + a2[R2] + a3[R3]
            - 0.12 * (logP_approx - 3.0)^2          (optimal lipophilicity)
            + 0.6  if R1 is a halogen and R2 is cyclopropyl   (pocket fit)
            - 1.0  if R2 is piperidinyl and R3 is Cl          (steric clash: activity cliff)
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .fragments import R1_FRAGMENTS, R2_FRAGMENTS, R3_FRAGMENTS, build_smiles, descriptors

TRUE_A1 = {"F": 0.3, "Cl": 0.7, "Br": 0.8, "Me": 0.2, "iPr": 0.1,
           "OMe": 0.1, "CF3": 0.9, "NMe2": -0.2, "CN": 0.4, "OH": -0.3}
TRUE_A2 = {"Me": 0.0, "Et": 0.2, "iPr": 0.4, "cPr": 0.8, "cHex": 0.5,
           "Ph": 0.3, "HOEt": -0.2, "Pip": 0.1}
TRUE_A3 = {"H": 0.0, "F": 0.2, "Cl": 0.3, "Me": 0.0, "OMe": -0.4}
HALOGENS = {"F", "Cl", "Br"}

TARGET_ID = "TGT-1"
CENSOR_CEILING_NM = 10_000.0  # assay upper limit: weaker compounds are reported as ">10000 nM"


def true_pic50(r1: str, r2: str, r3: str) -> float:
    """Ground-truth pIC50 for one compound (see module docstring)."""
    logp = descriptors(r1, r2, r3)["logp_approx"]
    value = 5.2 + TRUE_A1[r1] + TRUE_A2[r2] + TRUE_A3[r3] - 0.12 * (logp - 3.0) ** 2
    if r1 in HALOGENS and r2 == "cPr":
        value += 0.6
    if r2 == "Pip" and r3 == "Cl":
        value -= 1.0
    return value


def generate_library() -> pd.DataFrame:
    """Enumerate the full 10 x 8 x 5 = 400 compound library with descriptors."""
    rows = []
    combos = itertools.product(R1_FRAGMENTS, R2_FRAGMENTS, R3_FRAGMENTS)
    for i, (r1, r2, r3) in enumerate(combos, start=1):
        row = {
            "compound_id": f"CMPD-{i:04d}",
            "r1": r1,
            "r2": r2,
            "r3": r3,
            "smiles": build_smiles(r1, r2, r3),
            "true_pic50": round(true_pic50(r1, r2, r3), 4),
        }
        row.update(descriptors(r1, r2, r3))
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Raw assay records with realistic, logged data-quality problems
# --------------------------------------------------------------------------

UNIT_SPELLINGS = {
    "nM": ["nM"] * 6 + ["nm", "NM"],
    "uM": ["uM"] * 4 + ["µM", "um"],
    "M": ["M"],
}
UNIT_FACTOR_FROM_NM = {"nM": 1.0, "uM": 1e-3, "M": 1e-9}

ERROR_RATES = {
    "unit_mislabel": 0.03,
    "negative_value": 0.01,
    "zero_value": 0.005,
    "missing_value": 0.01,
    "exact_duplicate": 0.02,
    "invalid_smiles": 0.02,
    "smiles_id_conflict": 0.01,
    "outlier_replicate": 0.02,
    "wrong_endpoint": 0.03,
}


def _measurement_unit(rng: np.random.Generator) -> str:
    return rng.choice(["nM", "uM", "M"], p=[0.60, 0.35, 0.05])


def _corrupt_smiles(smiles: str, rng: np.random.Generator) -> str:
    mode = rng.integers(0, 3)
    if mode == 0:
        return smiles[:-2]  # truncated
    if mode == 1:
        return smiles.replace(")", "", 1)  # unbalanced parenthesis
    return smiles.replace("c1ccc", "c1cc", 1)  # broken aromatic ring


def generate_assay_records(library: pd.DataFrame, seed: int = 7):
    """Create raw assay records and a ground-truth injection log.

    Returns
    -------
    records : DataFrame
        ChEMBL-like raw table, including injected data-quality problems.
    injection_log : DataFrame
        ``record_id`` and ``error_type`` for every injected problem. This is the
        answer key used only to *evaluate* the QC pipeline.
    """
    rng = np.random.default_rng(seed)
    rows = []
    rec_no = 0
    for comp in library.itertuples():
        n_rep = rng.choice([1, 2, 3, 4], p=[0.30, 0.30, 0.25, 0.15])
        for _ in range(n_rep):
            rec_no += 1
            pic50 = comp.true_pic50 + rng.normal(0, 0.25)
            ic50_nm = 10 ** (9 - pic50)
            relation = "="
            if ic50_nm > CENSOR_CEILING_NM:
                relation, ic50_nm = ">", CENSOR_CEILING_NM
            unit = _measurement_unit(rng)
            value = ic50_nm * UNIT_FACTOR_FROM_NM[unit]
            rows.append(
                {
                    "record_id": f"REC-{rec_no:06d}",
                    "compound_id": comp.compound_id,
                    "smiles": comp.smiles,
                    "target_id": TARGET_ID,
                    "assay_id": rng.choice(["ASSAY-A", "ASSAY-B", "ASSAY-C"]),
                    "standard_type": "IC50",
                    "standard_relation": relation,
                    "standard_value": float(f"{value:.6g}"),
                    "standard_units": rng.choice(UNIT_SPELLINGS[unit]),
                    "date": str(pd.Timestamp("2018-01-01") + pd.Timedelta(days=int(rng.integers(0, 2500))))[:10],
                }
            )
    records = pd.DataFrame(rows)

    # ---- inject problems (each record gets at most one) -------------------
    log = []
    next_id = rec_no
    all_smiles = library["smiles"].to_numpy()
    dup_rows = []
    order = rng.permutation(len(records))
    cursor = 0
    for error_type, rate in ERROR_RATES.items():
        n_target = int(round(rate * len(records)))
        picked = 0
        while picked < n_target and cursor < len(order):
            idx = order[cursor]
            cursor += 1
            rec = records.iloc[idx]
            is_censored = rec["standard_relation"] != "="
            value_error = error_type in {
                "unit_mislabel", "negative_value", "zero_value",
                "missing_value", "outlier_replicate", "wrong_endpoint",
            }
            if is_censored and value_error:
                continue
            if error_type == "unit_mislabel":
                swap = {"nM": "uM", "uM": "nM"}
                base = {"nm": "nM", "NM": "nM", "µM": "uM", "um": "uM"}.get(
                    rec["standard_units"], rec["standard_units"]
                )
                if base not in swap:
                    continue
                records.at[idx, "standard_units"] = swap[base]
            elif error_type == "negative_value":
                records.at[idx, "standard_value"] = -abs(rec["standard_value"])
            elif error_type == "zero_value":
                records.at[idx, "standard_value"] = 0.0
            elif error_type == "missing_value":
                records.at[idx, "standard_value"] = np.nan
            elif error_type == "outlier_replicate":
                shift = rng.choice([-1, 1]) * rng.uniform(1.5, 2.5)
                records.at[idx, "standard_value"] = float(f"{rec['standard_value'] * 10 ** shift:.6g}")
            elif error_type == "wrong_endpoint":
                records.at[idx, "standard_type"] = rng.choice(["Ki", "EC50"])
            elif error_type == "invalid_smiles":
                records.at[idx, "smiles"] = _corrupt_smiles(rec["smiles"], rng)
            elif error_type == "smiles_id_conflict":
                other = rec["smiles"]
                while other == rec["smiles"]:
                    other = rng.choice(all_smiles)
                records.at[idx, "smiles"] = other
            elif error_type == "exact_duplicate":
                next_id += 1
                dup = rec.copy()
                dup["record_id"] = f"REC-{next_id:06d}"
                dup_rows.append(dup)
                log.append({"record_id": dup["record_id"], "error_type": error_type})
                picked += 1
                continue
            log.append({"record_id": rec["record_id"], "error_type": error_type})
            picked += 1

    if dup_rows:
        records = pd.concat([records, pd.DataFrame(dup_rows)], ignore_index=True)
    records = records.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return records, pd.DataFrame(log)
