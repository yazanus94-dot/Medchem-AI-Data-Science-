import numpy as np
import pandas as pd
import pytest

from bioqc.checks import (
    check_assay_range,
    check_exact_duplicates,
    check_missing_and_nonpositive,
    check_replicates,
    check_smiles,
    check_units,
    run_record_checks,
    smiles_is_balanced,
)
from medchem_common import build_smiles, parse_smiles


def _rec(**kw):
    base = {
        "record_id": "R1", "compound_id": "C1", "smiles": build_smiles("F", "Me", "H"),
        "target_id": "T", "assay_id": "A", "standard_type": "IC50",
        "standard_relation": "=", "standard_value": 100.0, "standard_units": "nM",
        "date": "2020-01-01",
    }
    base.update(kw)
    return base


def test_smiles_roundtrip_for_whole_library():
    from medchem_common.synthetic import generate_library

    lib = generate_library()
    assert len(lib) == 400
    assert all(parse_smiles(s) is not None for s in lib["smiles"])
    assert lib["smiles"].is_unique


@pytest.mark.parametrize("bad", ["O=C(NC)c1ccc(F)cc1)", "O=C(NC)c1ccc(F", "O=C(NC)c1ccc(F)cc", ""])
def test_unbalanced_smiles_rejected(bad):
    assert not smiles_is_balanced(bad)


def test_invalid_smiles_flagged():
    df = pd.DataFrame([_rec(), _rec(record_id="R2", smiles="O=C(NC)c1cc(F)cc1")])
    out = check_smiles(df)
    assert out["flag_invalid_smiles"].tolist() == [False, True]


def test_unit_normalisation_and_unknown_unit():
    df = pd.DataFrame([_rec(standard_units=u) for u in ["nM", "NM", "uM", "µM", "M", "mg/mL"]])
    out = check_units(df)
    assert out["unit_factor_to_nm"].tolist()[:5] == [1.0, 1.0, 1e3, 1e3, 1e9]
    assert out["flag_unknown_unit"].tolist() == [False] * 5 + [True]


def test_missing_zero_negative():
    df = pd.DataFrame([_rec(standard_value=v) for v in [np.nan, 0.0, -5.0, 5.0]])
    out = check_missing_and_nonpositive(df)
    assert out["flag_missing_value"].tolist() == [True, False, False, False]
    assert out["flag_nonpositive_value"].tolist() == [False, True, True, False]


def test_exact_duplicate_flags_only_later_copy():
    df = pd.DataFrame([_rec(), _rec(record_id="R2")])
    out = check_exact_duplicates(df)
    assert out["flag_exact_duplicate"].tolist() == [False, True]


def test_assay_range_flags_impossible_exact_value():
    df = pd.DataFrame([_rec(standard_value=5e4, record_id="R1"), _rec(standard_value=50.0, record_id="R2")])
    out = check_assay_range(run_record_checks(df).drop(columns=["flag_outside_assay_range"]))
    assert out["flag_outside_assay_range"].tolist() == [True, False]


def test_replicate_outlier_and_probable_unit_error():
    values = [5.0, 5.5, 4.75, 5.0 * 1000]  # last is exactly 3 log units off but inside the assay range
    df = pd.DataFrame([_rec(record_id=f"R{i}", standard_value=v, assay_id=f"A{i}") for i, v in enumerate(values)])
    out = check_replicates(run_record_checks(df))
    assert out["flag_probable_unit_error"].tolist() == [False, False, False, True]

    values = [100.0, 110.0, 95.0, 100.0 * 10 ** 1.8]
    df = pd.DataFrame([_rec(record_id=f"R{i}", standard_value=v, assay_id=f"A{i}") for i, v in enumerate(values)])
    out = check_replicates(run_record_checks(df))
    assert out["flag_replicate_outlier"].tolist() == [False, False, False, True]


def test_discordant_pair_flags_both():
    df = pd.DataFrame([_rec(record_id="R1", standard_value=100.0), _rec(record_id="R2", standard_value=5000.0)])
    out = check_replicates(run_record_checks(df))
    assert out["flag_discordant_pair"].tolist() == [True, True]


def test_concordant_pair_not_flagged():
    df = pd.DataFrame([_rec(record_id="R1", standard_value=100.0), _rec(record_id="R2", standard_value=140.0)])
    out = check_replicates(run_record_checks(df))
    assert not out["flag_discordant_pair"].any()
