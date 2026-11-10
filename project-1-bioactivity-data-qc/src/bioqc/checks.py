"""Record-level and replicate-level quality checks.

Every check adds a boolean ``flag_*`` column (True = problem found) and never
drops rows, so the full audit trail is preserved. Which flags block a record
from the curated dataset is decided in :mod:`bioqc.clean`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from medchem_common import parse_smiles

# Multiplicative factors that convert a reported concentration to nM.
UNIT_TO_NM = {
    "nm": 1.0,
    "um": 1e3,
    "µm": 1e3,  # micro sign
    "μm": 1e3,  # greek mu
    "m": 1e9,
}

RECORD_LEVEL_FLAGS = [
    "flag_missing_value",
    "flag_nonpositive_value",
    "flag_unknown_unit",
    "flag_endpoint_mismatch",
    "flag_invalid_smiles",
    "flag_exact_duplicate",
    "flag_structure_conflict",
    "flag_outside_assay_range",
]
REPLICATE_FLAGS = [
    "flag_probable_unit_error",
    "flag_replicate_outlier",
    "flag_discordant_pair",
]
ALL_FLAGS = RECORD_LEVEL_FLAGS + REPLICATE_FLAGS


def smiles_is_balanced(smiles: str) -> bool:
    """Generic syntax check: balanced parentheses and every ring digit paired."""
    if not isinstance(smiles, str) or not smiles:
        return False
    depth = 0
    open_rings: set[str] = set()
    for ch in smiles:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                return False
        elif ch.isdigit():
            open_rings ^= {ch}
    return depth == 0 and not open_rings


def check_missing_and_nonpositive(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["flag_missing_value"] = out["standard_value"].isna()
    out["flag_nonpositive_value"] = out["standard_value"].fillna(1.0) <= 0
    return out


def check_units(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise unit spelling and flag units that cannot be converted."""
    out = df.copy()
    key = out["standard_units"].astype("string").str.strip().str.lower()
    out["unit_factor_to_nm"] = key.map(UNIT_TO_NM).astype(float)
    out["flag_unknown_unit"] = out["unit_factor_to_nm"].isna()
    return out


def check_endpoint(df: pd.DataFrame, expected: str = "IC50") -> pd.DataFrame:
    out = df.copy()
    out["flag_endpoint_mismatch"] = out["standard_type"].astype("string").str.upper() != expected
    return out


def check_smiles(df: pd.DataFrame) -> pd.DataFrame:
    """A structure is valid if it is syntactically balanced AND belongs to the series.

    For arbitrary chemistry replace ``parse_smiles`` with RDKit's
    ``Chem.MolFromSmiles`` (returns ``None`` for invalid structures).
    """
    out = df.copy()
    balanced = out["smiles"].map(smiles_is_balanced)
    in_series = out["smiles"].map(lambda s: parse_smiles(s) is not None)
    out["flag_invalid_smiles"] = ~(balanced & in_series)
    return out


def check_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Flag later copies of records identical in every scientific field."""
    out = df.copy().sort_values("record_id")
    key = ["compound_id", "smiles", "target_id", "assay_id", "standard_type",
           "standard_relation", "standard_value", "standard_units", "date"]
    out["flag_exact_duplicate"] = out.duplicated(subset=key, keep="first")
    return out.sort_index()


def check_structure_conflicts(df: pd.DataFrame) -> pd.DataFrame:
    """Flag records whose structure disagrees with the rest of the same compound ID.

    The reference structure is the most frequent *valid* SMILES per compound.
    If the top two are tied, every record in the conflicting group is flagged,
    because there is no basis to decide which one is right. A compound with a
    single record cannot be checked this way (known limitation).
    """
    out = df.copy()
    out["flag_structure_conflict"] = False
    valid = out[~out["flag_invalid_smiles"]]
    for _, grp in valid.groupby("compound_id"):
        counts = grp["smiles"].value_counts()
        if len(counts) < 2:
            continue
        tied = len(counts) > 1 and counts.iloc[0] == counts.iloc[1]
        if tied:
            out.loc[grp.index, "flag_structure_conflict"] = True
        else:
            out.loc[grp.index[grp["smiles"] != counts.index[0]], "flag_structure_conflict"] = True
    return out


def add_potency(df: pd.DataFrame) -> pd.DataFrame:
    """Compute IC50 in nM and pIC50 for rows with a usable value and unit."""
    out = df.copy()
    usable = (
        ~out["flag_missing_value"]
        & ~out["flag_nonpositive_value"]
        & ~out["flag_unknown_unit"]
    )
    out["ic50_nm"] = np.where(usable, out["standard_value"] * out["unit_factor_to_nm"], np.nan)
    out["pic50"] = np.where(usable, 9.0 - np.log10(out["ic50_nm"].where(usable, 1.0)), np.nan)
    out["is_censored"] = out["standard_relation"].astype("string").str.strip() != "="
    return out


def check_assay_range(
    df: pd.DataFrame, lower_nm: float = 0.1, upper_nm: float = 10_000.0
) -> pd.DataFrame:
    """Flag exact ("=") results outside the assay's dynamic range.

    An assay with a 10 uM ceiling cannot report an exact IC50 of 1 mM; such a
    value points to a unit or transcription error. The range is an assumption
    taken from the assay protocol and is configurable.
    """
    out = df.copy()
    exact = ~out["is_censored"] & out["ic50_nm"].notna()
    out["flag_outside_assay_range"] = exact & (
        (out["ic50_nm"] > upper_nm) | (out["ic50_nm"] < lower_nm)
    )
    return out


def check_replicates(
    df: pd.DataFrame,
    outlier_threshold: float = 3.0,
    sigma_floor: float = 0.2,
    discordant_log_units: float = 1.0,
) -> pd.DataFrame:
    """Replicate-level checks on records that passed all record-level checks.

    * >= 3 replicates: robust z-score (median / MAD). A record more than
      ``outlier_threshold`` robust SDs from the median is an outlier. If its
      deviation is within 0.3 log units of +/-3 it is labelled a probable
      nM/uM unit error instead (a 1000-fold shift is exactly 3 log units).
    * exactly 2 replicates: no majority exists, so a pair that disagrees by more
      than ``discordant_log_units`` is flagged as discordant (both records).
    * 1 replicate: cannot be checked (known limitation).
    """
    out = df.copy()
    for col in REPLICATE_FLAGS:
        out[col] = False
    record_flags = [c for c in RECORD_LEVEL_FLAGS if c in out.columns]
    eligible = out[~out[record_flags].any(axis=1) & ~out["is_censored"] & out["pic50"].notna()]
    for _, grp in eligible.groupby("compound_id"):
        x = grp["pic50"]
        if len(grp) >= 3:
            med = x.median()
            sigma = max(1.4826 * (x - med).abs().median(), sigma_floor)
            dev = x - med
            bad = dev.abs() > outlier_threshold * sigma
            near_three = (dev.abs() - 3.0).abs() <= 0.3
            out.loc[grp.index[bad & near_three], "flag_probable_unit_error"] = True
            out.loc[grp.index[bad & ~near_three], "flag_replicate_outlier"] = True
        elif len(grp) == 2:
            if abs(x.iloc[0] - x.iloc[1]) > discordant_log_units:
                out.loc[grp.index, "flag_discordant_pair"] = True
    return out


def run_record_checks(df: pd.DataFrame) -> pd.DataFrame:
    """Run all record-level checks and add potency columns."""
    out = check_missing_and_nonpositive(df)
    out = check_units(out)
    out = check_endpoint(out)
    out = check_smiles(out)
    out = check_exact_duplicates(out)
    out = check_structure_conflicts(out)
    out = add_potency(out)
    return check_assay_range(out)
