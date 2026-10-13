"""Elastic net via coordinate descent (REG-03, REG-06) — Component 6 only."""

from __future__ import annotations

import numpy as np


def _soft_threshold(x: float, t: float) -> float:
    if x > t:
        return x - t
    if x < -t:
        return x + t
    return 0.0


def elastic_net_fit(X: np.ndarray, y: np.ndarray, lam: float, l1_ratio: float = 0.5,
                    max_iter: int = 1000, tol: float = 1e-6) -> np.ndarray:
    """Sparse-with-correlated-groups fit; returns coefficients in standardized X space."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape
    sd = X.std(0)
    Z = (X - X.mean(0)) / np.where(sd > 1e-12, sd, 1.0)
    r = y - y.mean()
    beta = np.zeros(p)
    l1, l2 = lam * l1_ratio, lam * (1 - l1_ratio)
    col_sq = np.sum(Z ** 2, axis=0) / n
    for _ in range(max_iter):
        beta_old = beta.copy()
        for j in range(p):
            if col_sq[j] == 0:
                continue
            r = r + Z[:, j] * beta[j]
            beta[j] = _soft_threshold(float(Z[:, j] @ r) / n, l1) / (col_sq[j] + l2)
            r = r - Z[:, j] * beta[j]
        if np.max(np.abs(beta - beta_old)) < tol:
            break
    return beta
