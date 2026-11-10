"""Training and prospective sets for the synthetic benzamide series."""

from __future__ import annotations

import numpy as np
import pandas as pd

from medchem_common.synthetic import generate_library

PHYSCHEM = ["mw", "logp_approx", "hbd", "hba", "tpsa_approx", "heavy_atoms"]


def make_dataset(seed: int = 11, n_measured: int = 300, noise_sd: float = 0.3):
    """Split the 400-compound library into a measured set and a virtual set.

    * ``measured``: ``n_measured`` randomly chosen compounds with one noisy
      potency each (true pIC50 + Gaussian noise, SD ``noise_sd``), as would come
      out of a curated screening campaign.
    * ``virtual``: the remaining compounds, whose true pIC50 is hidden from the
      models and used only to judge prospective ranking.

    Weak compounds are *not* censored here (see project 1 for censoring); this
    project isolates the modelling question.
    """
    library = generate_library()
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(library))
    measured = library.iloc[idx[:n_measured]].copy().reset_index(drop=True)
    virtual = library.iloc[idx[n_measured:]].copy().reset_index(drop=True)
    measured["pic50"] = measured["true_pic50"] + rng.normal(0, noise_sd, len(measured))
    return library, measured, virtual
