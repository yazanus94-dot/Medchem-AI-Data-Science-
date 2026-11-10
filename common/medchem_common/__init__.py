"""Shared helpers for the three portfolio projects.

The package provides a small, fully transparent chemistry toolkit for a
synthetic benzamide series (fragment tables, additive property estimates, a
template-based SMILES builder/parser) and a generator for synthetic bioactivity
data with a known ground-truth SAR.
"""

from .fragments import (
    R1_FRAGMENTS,
    R2_FRAGMENTS,
    R3_FRAGMENTS,
    build_smiles,
    descriptors,
    formula_string,
    parse_smiles,
)

__all__ = [
    "R1_FRAGMENTS",
    "R2_FRAGMENTS",
    "R3_FRAGMENTS",
    "build_smiles",
    "descriptors",
    "formula_string",
    "parse_smiles",
]
