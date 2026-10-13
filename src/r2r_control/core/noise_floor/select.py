"""Floor selection: analytic vs empirical, with the balanced-panel check (NF-02/04)."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .analytic import analytic_floor, NoiseFloorResult
from .empirical import empirical_floor


def estimate(data: np.ndarray, missing_mask: Optional[np.ndarray] = None,
             n_perm: int = 500, quantile: float = 0.99,
             rng: Optional[np.random.Generator] = None,
             balanced_tolerance: float = 1e-9,
             surrogate: str = "block") -> NoiseFloorResult:
    """Choose analytic vs empirical floor by checking the balanced-panel assumption.

    Balanced panel (no missingness) => analytic MP edge. Otherwise — the usual
    case under sparse/asynchronous metrology (§DM-02) — the structure-preserving
    empirical floor takes precedence and the reason is recorded (NF-02/04).
    """
    X = np.asarray(data, dtype=float)
    T, N = X.shape
    if missing_mask is None:
        missing_mask = ~np.isfinite(X)
    frac_missing = float(missing_mask.mean())
    if frac_missing <= balanced_tolerance:
        res = analytic_floor(N, T, variance=1.0)
        res.detail["frac_missing"] = frac_missing
        return res
    res = empirical_floor(X, missing_mask, n_perm=n_perm, quantile=quantile, rng=rng,
                          surrogate=surrogate)
    res.reason += (f"; analytic MP refused because panel is unbalanced "
                   f"(frac_missing={frac_missing:.3f}, balanced-panel assumption "
                   f"violated, NF-02)")
    res.detail["frac_missing"] = frac_missing
    return res
