"""Fragment tables and additive property calculators for a benzamide library.

The synthetic series used throughout this portfolio is a para-substituted
N-substituted benzamide with an optional meta substituent::

    O=C(N-R2)c1ccc(R1)c(R3)c1        (R3 = H gives ...c(R1)cc1)

Because the library is built from a fixed template, molecular formula, molecular
weight and the other descriptors can be computed *exactly from a lookup table*
without a cheminformatics toolkit.

IMPORTANT - scope of the property estimates
-------------------------------------------
* Molecular weight, heavy-atom count, H-bond donors/acceptors follow directly
  from the formula and are exact for this series.
* ``logp_approx`` is a Hansch-style *additive approximation* (benzamide base value
  plus substituent pi constants). It is illustrative, not a validated predictor.
* ``tpsa_approx`` sums Ertl-style polar-fragment contributions that apply to
  these fragments.

For real compound sets, use a toolkit such as RDKit for parsing, standardisation
and descriptors. This module exists so the projects run with only NumPy, pandas
and scikit-learn, and so every number can be traced by hand.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

ATOMIC_MASS = {
    "C": 12.011,
    "H": 1.008,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998,
    "Cl": 35.45,
    "Br": 79.904,
}

BASE_LOGP = 0.64  # benzamide
AMIDE_TPSA = 29.10  # secondary amide: N-H (12.03) + C=O (17.07)


@dataclass(frozen=True)
class Fragment:
    """A substituent: SMILES token, atoms it contributes and simple properties."""

    name: str
    smiles: str
    atoms: dict = field(default_factory=dict)
    pi: float = 0.0  # approximate lipophilicity contribution
    hbd: int = 0  # extra H-bond donors beyond the amide N-H
    tpsa: float = 0.0  # extra polar surface area


def _f(name, smiles, atoms, pi, hbd=0, tpsa=0.0):
    return Fragment(name, smiles, atoms, pi, hbd, tpsa)


# Para substituent (always present) ---------------------------------------
R1_FRAGMENTS = {
    f.name: f
    for f in [
        _f("F", "F", {"F": 1}, 0.14),
        _f("Cl", "Cl", {"Cl": 1}, 0.71),
        _f("Br", "Br", {"Br": 1}, 0.86),
        _f("Me", "C", {"C": 1, "H": 3}, 0.56),
        _f("iPr", "C(C)C", {"C": 3, "H": 7}, 1.53),
        _f("OMe", "OC", {"O": 1, "C": 1, "H": 3}, -0.02, 0, 9.23),
        _f("CF3", "C(F)(F)F", {"C": 1, "F": 3}, 0.88),
        _f("NMe2", "N(C)C", {"N": 1, "C": 2, "H": 6}, 0.18, 0, 3.24),
        _f("CN", "C#N", {"C": 1, "N": 1}, -0.57, 0, 23.79),
        _f("OH", "O", {"O": 1, "H": 1}, -0.67, 1, 20.23),
    ]
}

# Amide N substituent -------------------------------------------------------
R2_FRAGMENTS = {
    f.name: f
    for f in [
        _f("Me", "C", {"C": 1, "H": 3}, 0.22),
        _f("Et", "CC", {"C": 2, "H": 5}, 0.70),
        _f("iPr", "C(C)C", {"C": 3, "H": 7}, 1.00),
        _f("cPr", "C1CC1", {"C": 3, "H": 5}, 0.70),
        _f("cHex", "C1CCCCC1", {"C": 6, "H": 11}, 2.00),
        _f("Ph", "c1ccccc1", {"C": 6, "H": 5}, 2.00),
        _f("HOEt", "CCO", {"C": 2, "H": 5, "O": 1}, -0.45, 1, 20.23),
        _f("Pip", "C1CCNCC1", {"C": 5, "H": 10, "N": 1}, -0.20, 1, 12.03),
    ]
}

# Optional meta substituent (None = hydrogen) -------------------------------
R3_FRAGMENTS = {
    "H": None,
    **{
        name: R1_FRAGMENTS[name]
        for name in ["F", "Cl", "Me", "OMe"]
    },
}


def _add(total: dict, atoms: dict, times: int = 1) -> None:
    for el, n in atoms.items():
        total[el] = total.get(el, 0) + n * times


def composition(r1: str, r2: str, r3: str = "H") -> dict:
    """Atom counts for the molecule defined by the three substituent names."""
    # Core: aromatic ring C6, carbonyl C, amide N and O, amide N-H.
    aromatic_h = 6 - 1 - 1 - (0 if r3 == "H" else 1)  # minus C(=O), minus R1, minus R3
    total = {"C": 7, "N": 1, "O": 1, "H": aromatic_h + 1}
    _add(total, R1_FRAGMENTS[r1].atoms)
    _add(total, R2_FRAGMENTS[r2].atoms)
    if r3 != "H":
        _add(total, R3_FRAGMENTS[r3].atoms)
    return total


def formula_string(r1: str, r2: str, r3: str = "H") -> str:
    """Molecular formula in Hill order (C, H, then alphabetical)."""
    comp = composition(r1, r2, r3)
    order = ["C", "H"] + sorted(k for k in comp if k not in ("C", "H"))
    return "".join(f"{el}{comp[el] if comp[el] > 1 else ''}" for el in order if el in comp)


def descriptors(r1: str, r2: str, r3: str = "H") -> dict:
    """Exact and approximate descriptors for one compound (see module docstring)."""
    comp = composition(r1, r2, r3)
    mw = sum(ATOMIC_MASS[el] * n for el, n in comp.items())
    heavy = sum(n for el, n in comp.items() if el != "H")
    frag3 = R3_FRAGMENTS[r3]
    logp = BASE_LOGP + R1_FRAGMENTS[r1].pi + R2_FRAGMENTS[r2].pi
    tpsa = AMIDE_TPSA + R1_FRAGMENTS[r1].tpsa + R2_FRAGMENTS[r2].tpsa
    hbd = 1 + R1_FRAGMENTS[r1].hbd + R2_FRAGMENTS[r2].hbd
    if frag3 is not None:
        logp += frag3.pi
        tpsa += frag3.tpsa
        hbd += frag3.hbd
    return {
        "formula": formula_string(r1, r2, r3),
        "mw": round(mw, 3),
        "heavy_atoms": heavy,
        "logp_approx": round(logp, 2),
        "hbd": hbd,
        "hba": comp.get("N", 0) + comp.get("O", 0),
        "tpsa_approx": round(tpsa, 2),
    }


def build_smiles(r1: str, r2: str, r3: str = "H") -> str:
    """SMILES string for the template with the given substituents."""
    s1 = R1_FRAGMENTS[r1].smiles
    s2 = R2_FRAGMENTS[r2].smiles
    if r3 == "H":
        return f"O=C(N{s2})c1ccc({s1})cc1"
    return f"O=C(N{s2})c1ccc({s1})c({R3_FRAGMENTS[r3].smiles})c1"


_TEMPLATE = re.compile(
    r"^O=C\(N(?P<r2>.+?)\)c1ccc\((?P<r1>.+?)\)(?:cc1|c\((?P<r3>.+?)\)c1)$"
)
_R1_BY_SMILES = {f.smiles: n for n, f in R1_FRAGMENTS.items()}
_R2_BY_SMILES = {f.smiles: n for n, f in R2_FRAGMENTS.items()}
_R3_BY_SMILES = {f.smiles: n for n, f in R3_FRAGMENTS.items() if f is not None}


def parse_smiles(smiles: str):
    """Return ``(r1, r2, r3)`` if the SMILES belongs to the library, else ``None``.

    The check is deliberately strict: the string is decomposed with the library
    template and then rebuilt; it is valid only if the rebuilt string is identical.
    """
    if not isinstance(smiles, str):
        return None
    match = _TEMPLATE.match(smiles.strip())
    if not match:
        return None
    r1 = _R1_BY_SMILES.get(match.group("r1"))
    r2 = _R2_BY_SMILES.get(match.group("r2"))
    r3_smiles = match.group("r3")
    r3 = "H" if r3_smiles is None else _R3_BY_SMILES.get(r3_smiles)
    if r1 is None or r2 is None or r3 is None:
        return None
    if build_smiles(r1, r2, r3) != smiles.strip():
        return None
    return r1, r2, r3
