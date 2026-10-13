"""DIAG-04 — variance decomposition into recognized sources."""

from __future__ import annotations

from typing import Dict

import numpy as np

from ...core.noise_floor.analytic import minimum_variance_benchmark


def variance_decomposition(y_observed: np.ndarray, ff_leakage: Dict, gain_mismatch: Dict,
                           qr: Dict, delay: int = 1) -> Dict:
    """Split residual variance into FF / gain / estimator / irreducible (DIAG-04).

    Only causes the precedence tree actually established get a share; anything
    recoverable but unattributed (including shares blocked by an
    "unidentifiable" node) stays in ``unattributed_recoverable``. The FB-08
    caveat is surfaced, not hidden.
    """
    Y = np.asarray(y_observed, dtype=float)
    total_var = float(np.nanmean(np.nanvar(Y, axis=0)))
    mv = np.nanmean([minimum_variance_benchmark(Y[:, j], delay) for j in range(Y.shape[1])])
    irreducible = float(np.clip(mv / (total_var + 1e-12), 0, 1))
    recoverable = 1.0 - irreducible

    w = {"incorrect_ff_model": 1.0 if ff_leakage.get("leaking_inputs") else 0.0,
         "incorrect_process_gain": 1.0 if (gain_mismatch.get("gain_underestimated") is True
                                           or gain_mismatch.get("oscillating") is True) else 0.0,
         "suboptimal_estimator_tuning": 1.0 if qr.get("innovation_colored") is True else 0.0}
    w_sum = sum(w.values())
    shares = {k: (recoverable * v / w_sum if w_sum else 0.0) for k, v in w.items()}
    shares["irreducible_noise"] = irreducible
    shares["unattributed_recoverable"] = 0.0 if w_sum else recoverable
    return {"total_variance": total_var, "shares": shares,
            "identifiability_caveat": ("gain vs estimator-tuning shares are not separable "
                                       "from passive data alone (FB-08)")}
