"""Structure-preserving empirical noise floor (NF-03, E6).

A plain time-shuffle SHALL NOT be used: it destroys autocorrelation as well as
cross-correlation, so under autocorrelated/drifting data it *understates* the
floor — autocorrelated noise produces larger eigenvalues than its shuffled
version and would be misread as signal. Instead we use a structure-preserving
surrogate (block permutation by default, or phase randomization) that destroys
cross-series correlation while preserving each series' marginal distribution,
autocorrelation, and exact missingness pattern. This is the spectral
counterpart of the block-bootstrap rule (§VAL-05) and shares its block logic.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .analytic import NoiseFloorResult
from ..validation.block_bootstrap import contiguous_block_permutation


def _phase_surrogate(vals: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Phase-randomized surrogate: keeps the power spectrum (hence autocorrelation)."""
    m = len(vals)
    mu, sd = vals.mean(), vals.std()
    F = np.fft.rfft(vals - mu)
    phases = np.exp(1j * rng.uniform(0, 2 * np.pi, size=F.shape))
    phases[0] = 1.0
    if m % 2 == 0:
        phases[-1] = 1.0
    surr = np.fft.irfft(np.abs(F) * phases, n=m)
    s = surr.std()
    if s > 1e-12:
        surr = surr / s * sd
    return surr + mu


def _top_corr_eigenvalue(mat: np.ndarray) -> float:
    T = mat.shape[0]
    Z = np.zeros_like(mat)
    for j in range(mat.shape[1]):
        col = mat[:, j]
        obs = np.isfinite(col)
        if obs.sum() < 2:
            continue
        sd = col[obs].std()
        if sd < 1e-12:
            continue
        Z[obs, j] = (col[obs] - col[obs].mean()) / sd
    C = (Z.T @ Z) / max(T - 1, 1)
    return float(np.linalg.eigvalsh(C)[-1])


def empirical_floor(data: np.ndarray, missing_mask: Optional[np.ndarray] = None,
                    n_perm: int = 500, quantile: float = 0.99,
                    rng: Optional[np.random.Generator] = None,
                    surrogate: str = "block",
                    block_size: Optional[int] = None) -> NoiseFloorResult:
    """Structure-preserving surrogate noise floor (NF-03).

    ``surrogate`` is ``"block"`` (contiguous block permutation, default) or
    ``"phase"`` (phase randomization). Both preserve each column's marginal and
    autocorrelation and re-apply the exact missingness pattern. The ``quantile``
    of the null distribution of the top eigenvalue is the floor.
    """
    if surrogate not in ("block", "phase"):
        raise ValueError(f"unknown surrogate {surrogate!r}; a plain shuffle is "
                         f"forbidden (NF-03)")
    if rng is None:
        rng = np.random.default_rng(0)
    X = np.asarray(data, dtype=float)
    T, N = X.shape
    if missing_mask is None:
        missing_mask = ~np.isfinite(X)
    observed = ~missing_mask

    null_stats = np.empty(n_perm)
    for k in range(n_perm):
        Xp = np.full_like(X, np.nan)
        for j in range(N):
            obs_idx = np.where(observed[:, j])[0]
            if len(obs_idx) == 0:
                continue
            vals = X[obs_idx, j]
            if surrogate == "phase" and len(vals) >= 8:
                Xp[obs_idx, j] = _phase_surrogate(vals, rng)
            else:
                bs = block_size or max(2, int(round(np.sqrt(len(vals)))))
                Xp[obs_idx, j] = contiguous_block_permutation(vals, bs, rng)
        null_stats[k] = _top_corr_eigenvalue(Xp)
    return NoiseFloorResult(
        floor=float(np.quantile(null_stats, quantile)),
        kind=f"empirical_{surrogate}_surrogate",
        reason=(f"{int(quantile * 100)}th pct of {surrogate}-surrogate null of the top "
                f"eigenvalue; preserves marginal + autocorrelation + missingness "
                f"(NF-03; plain time-shuffle forbidden)"),
        assumptions_hold=True,
        detail={"n_perm": n_perm, "quantile": quantile, "surrogate": surrogate,
                "null_mean": float(null_stats.mean()),
                "null_max": float(null_stats.max())},
    )
