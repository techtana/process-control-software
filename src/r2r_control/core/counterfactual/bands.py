"""Monte-Carlo uncertainty bands on the counterfactual (CF-02, CF-03)."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ...contracts.shapes import check_shape_contract
from .reconstruct import reconstruct, CounterfactualResult


def reconstruct_with_bands(y_observed: np.ndarray, M: np.ndarray, u_used: np.ndarray,
                           u0: np.ndarray, rel_uncertainty_M: float = 0.0,
                           u0_uncertainty: Optional[np.ndarray] = None,
                           n_samples: int = 200,
                           rng: Optional[np.random.Generator] = None,
                           quantiles=(0.05, 0.5, 0.95),
                           u0_source: str = "unspecified") -> CounterfactualResult:
    """Monte-Carlo bands from relative ``M`` uncertainty and optional ``u0`` uncertainty."""
    if rng is None:
        rng = np.random.default_rng(0)
    y_observed = np.asarray(y_observed, dtype=float)
    u_used = np.asarray(u_used, dtype=float)
    u0 = np.asarray(u0, dtype=float).reshape(-1)
    n_out, n_knob = np.asarray(M).shape
    check_shape_contract(M, n_out, n_knob, where="counterfactual M")

    base = reconstruct(y_observed, M, u_used, u0, u0_source, n_out, n_knob)
    if rel_uncertainty_M <= 0 and u0_uncertainty is None:
        return base

    samples = np.empty((n_samples, *base.y_nocontrol.shape))
    for s in range(n_samples):
        M_s = M * (1.0 + rel_uncertainty_M * rng.standard_normal(M.shape))
        u0_s = u0 if u0_uncertainty is None else \
            u0 + np.asarray(u0_uncertainty, dtype=float) * rng.standard_normal(u0.shape)
        samples[s] = y_observed - (u_used - u0_s[None, :]) @ M_s.T
    qs = np.quantile(samples, quantiles, axis=0)
    base.bands = {"quantiles": np.asarray(quantiles), "lower": qs[0],
                  "median": qs[len(quantiles) // 2], "upper": qs[-1],
                  "std": samples.std(axis=0), "rel_uncertainty_M": rel_uncertainty_M}
    base.u0_uncertainty = u0_uncertainty
    return base
