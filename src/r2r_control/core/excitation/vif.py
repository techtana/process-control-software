"""Variance-inflation factor — the stakeholder-facing confounding measure (EX-01)."""

from __future__ import annotations

import numpy as np


def _standardize(X: np.ndarray):
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd_safe = np.where(sd > 1e-12, sd, 1.0)
    Z = (X - mu) / sd_safe
    return Z, mu, sd


def vif(X: np.ndarray) -> np.ndarray:
    """Variance-inflation factor per regressor (EX-01).

    ``VIF_j = 1 / (1 - R_j^2)`` where ``R_j^2`` is from regressing column j on
    the others. Large VIF => that knob/FF direction is largely explained by the
    others, i.e. its apparent relationship to the output is confounded by
    feedback.
    """
    Z, _, sd = _standardize(np.asarray(X, dtype=float))
    n, p = Z.shape
    out = np.full(p, np.nan)
    for j in range(p):
        if sd[j] <= 1e-12:
            out[j] = np.inf
            continue
        others = np.delete(Z, j, axis=1)
        if others.shape[1] == 0:
            out[j] = 1.0
            continue
        beta, *_ = np.linalg.lstsq(others, Z[:, j], rcond=None)
        resid = Z[:, j] - others @ beta
        ss_res = float(resid @ resid)
        ss_tot = float(Z[:, j] @ Z[:, j])
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        r2 = min(max(r2, 0.0), 1.0 - 1e-12)
        out[j] = 1.0 / (1.0 - r2)
    return out
