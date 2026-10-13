"""Partial least squares via NIPALS (REG-03) — Component 6 only."""

from __future__ import annotations

import numpy as np


def pls_fit(X: np.ndarray, Y: np.ndarray, n_components: int):
    """Supervised latent components maximizing covariance with the target (REG-03).

    Returns ``B`` mapping centered X to centered Y, the scores, and ``W*``.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float).reshape(len(X), -1)
    n, p = X.shape
    m = Y.shape[1]
    Xr = X - X.mean(0)
    Yr = Y - Y.mean(0)
    n_components = max(1, min(n_components, min(n - 1, p)))
    W = np.zeros((p, n_components))
    P = np.zeros((p, n_components))
    Q = np.zeros((m, n_components))
    T = np.zeros((n, n_components))
    for a in range(n_components):
        u, _, _ = np.linalg.svd(Xr.T @ Yr, full_matrices=False)
        w = u[:, 0] / (np.linalg.norm(u[:, 0]) + 1e-15)
        t = Xr @ w
        tt = float(t @ t) + 1e-15
        pld = Xr.T @ t / tt
        q = Yr.T @ t / tt
        Xr = Xr - np.outer(t, pld)
        Yr = Yr - np.outer(t, q)
        W[:, a], P[:, a], Q[:, a], T[:, a] = w, pld, q, t
    Wstar = W @ np.linalg.pinv(P.T @ W)
    return Wstar @ Q.T, T, Wstar
