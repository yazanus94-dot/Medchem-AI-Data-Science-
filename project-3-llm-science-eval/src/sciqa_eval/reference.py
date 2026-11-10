"""Reference calculations for medicinal chemistry / bioinformatics questions.

Every benchmark answer is *computed* by one of these small, documented
functions rather than typed in by hand, so the answer key can be unit-tested.
"""

from __future__ import annotations

import math

from scipy.stats import hypergeom

R_KCAL = 0.001987204  # gas constant, kcal / (mol K)

ATOMIC_MASS = {"C": 12.011, "H": 1.008, "N": 14.007, "O": 15.999, "Cl": 35.45}


def pic50_from_ic50_nm(ic50_nm: float) -> float:
    """pIC50 = -log10(IC50 in mol/L) = 9 - log10(IC50 in nM)."""
    return 9.0 - math.log10(ic50_nm)


def ic50_nm_from_pic50(pic50: float) -> float:
    return 10 ** (9.0 - pic50)


def mean_pic50_from_ic50_nm(values_nm: list[float]) -> float:
    """Average potency in log space (the appropriate average for IC50 replicates)."""
    return sum(pic50_from_ic50_nm(v) for v in values_nm) / len(values_nm)


def cheng_prusoff_ki(ic50: float, substrate: float, km: float) -> float:
    """Ki = IC50 / (1 + [S]/Km) for a competitive inhibitor (same units in, same out)."""
    return ic50 / (1.0 + substrate / km)


def z_prime(mean_pos: float, sd_pos: float, mean_neg: float, sd_neg: float) -> float:
    """Z' = 1 - 3 (SD_pos + SD_neg) / |mean_pos - mean_neg|."""
    return 1.0 - 3.0 * (sd_pos + sd_neg) / abs(mean_pos - mean_neg)


def ligand_efficiency(pic50: float, heavy_atoms: int) -> float:
    """LE = 1.37 * pIC50 / heavy atoms  (kcal/mol per heavy atom, ~298 K)."""
    return 1.37 * pic50 / heavy_atoms


def lipophilic_ligand_efficiency(pic50: float, logp: float) -> float:
    """LLE = pIC50 - logP."""
    return pic50 - logp


def log2_fold_change(treated: float, control: float) -> float:
    return math.log2(treated / control)


def benjamini_hochberg(pvalues: list[float]) -> list[float]:
    """BH-adjusted p-values, returned in the input order."""
    n = len(pvalues)
    order = sorted(range(n), key=lambda i: pvalues[i])
    adjusted = [0.0] * n
    running = 1.0
    for rank in range(n, 0, -1):
        i = order[rank - 1]
        running = min(running, pvalues[i] * n / rank)
        adjusted[i] = running
    return adjusted


def bonferroni_threshold(alpha: float, n_tests: int) -> float:
    return alpha / n_tests


def hypergeometric_enrichment_p(population: int, pathway: int, hits: int, overlap: int) -> float:
    """P(X >= overlap) when drawing ``hits`` genes from ``population`` with ``pathway`` successes."""
    return float(hypergeom.sf(overlap - 1, population, pathway, hits))


def delta_g_binding_kcal(kd_molar: float, temperature_k: float = 298.15) -> float:
    """Standard binding free energy dG = RT ln(Kd)  (negative for tight binding)."""
    return R_KCAL * temperature_k * math.log(kd_molar)


def fractional_occupancy(ligand: float, kd: float) -> float:
    return ligand / (ligand + kd)


def stock_volume(stock_conc: float, final_conc: float, final_volume: float) -> float:
    """C1 V1 = C2 V2  ->  V1 = C2 V2 / C1 (consistent concentration units)."""
    return final_conc * final_volume / stock_conc


def molecular_weight(composition: dict[str, int]) -> float:
    return sum(ATOMIC_MASS[el] * n for el, n in composition.items())


def lipinski_violations(mw: float, logp: float, hbd: int, hba: int) -> int:
    """Number of rule-of-five criteria violated (MW > 500, logP > 5, HBD > 5, HBA > 10)."""
    return int(mw > 500) + int(logp > 5) + int(hbd > 5) + int(hba > 10)


def hill_fractional_inhibition(conc_over_ic50: float, hill: float = 1.0) -> float:
    x = conc_over_ic50 ** hill
    return x / (1.0 + x)


def selectivity_fold(off_target_ic50: float, on_target_ic50: float) -> float:
    return off_target_ic50 / on_target_ic50
