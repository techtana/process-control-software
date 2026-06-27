"""Noise-floor estimation (§4.4, NF-01..05).

The single answer to "where is the floor below which variation is
indistinguishable from noise?".  Provided two ways:

* **Analytic** — the Marchenko-Pastur edge ``(1 + sqrt(q))^2`` with ``q = N/T``
  for a *balanced* covariance panel, and the minimum-variance / Harris benchmark
  for achievable control residual.  Declares its assumptions and refuses to
  apply silently when they are violated (NF-02).
* **Empirical** — a permutation procedure that destroys cross-series
  correlation while preserving each series' marginal distribution and its exact
  missingness pattern, building the null distribution of the top eigenvalue; a
  high quantile of that null is the floor (NF-03).

When the analytic assumptions are violated the empirical floor takes precedence
and the choice is recorded (NF-04).  Optional eigenvalue cleaning (clipping and
shrinkage) is available for stabilizing identification covariances (NF-05).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Analytic floor (NF-01/02)
# --------------------------------------------------------------------------- #
def marchenko_pastur_edge(q: float) -> Tuple[float, float]:
    """Lower/upper edges of the MP bulk for aspect ratio ``q = N/T`` (NF-01).

    Eigenvalues of a sample correlation matrix of pure noise fall (asymptotically)
    inside ``[(1-sqrt(q))^2, (1+sqrt(q))^2]``.  The upper edge is the noise floor:
    an eigen-direction above it is signal.
    """
    sq = np.sqrt(q)
    return (1.0 - sq) ** 2, (1.0 + sq) ** 2


@dataclass
class NoiseFloorResult:
    floor: float
    kind: str                     # 'analytic_mp' | 'empirical_permutation'
    reason: str
    assumptions_hold: bool
    detail: dict

    def to_dict(self):
        return {
            "floor": float(self.floor),
            "kind": self.kind,
            "reason": self.reason,
            "assumptions_hold": self.assumptions_hold,
            "detail": self.detail,
        }


def analytic_floor(N: int, T: int, variance: float = 1.0) -> NoiseFloorResult:
    """MP analytic floor, with an explicit balanced-panel assumption (NF-01/02).

    ``q = N/T`` assumes a *balanced* panel — every one of the ``N`` series
    observed over all ``T`` periods.  This function returns the floor together
    with whether that assumption can be presumed; callers must pass a real
    balanced count or use :func:`estimate` which checks missingness.
    """
    q = N / T
    _, upper = marchenko_pastur_edge(q)
    floor = variance * upper
    return NoiseFloorResult(
        floor=float(floor),
        kind="analytic_mp",
        reason="balanced-panel MP edge (1+sqrt(N/T))^2",
        assumptions_hold=True,
        detail={"q": q, "upper_edge": upper, "variance": variance},
    )


# --------------------------------------------------------------------------- #
# Empirical floor (NF-03)
# --------------------------------------------------------------------------- #
def empirical_floor(
    data: np.ndarray,
    missing_mask: Optional[np.ndarray] = None,
    n_perm: int = 500,
    quantile: float = 0.99,
    rng: Optional[np.random.Generator] = None,
    statistic: str = "top_eigenvalue",
) -> NoiseFloorResult:
    """Permutation noise floor for assumption-violating data (NF-03).

    Each column (series) is independently shuffled, which destroys cross-series
    correlation while preserving the column's marginal distribution and — by
    shuffling only among the *observed* positions — its exact missingness
    pattern.  The top eigenvalue of the standardized correlation matrix is
    recomputed each permutation to build the null; the ``quantile`` of that null
    is the floor.  Real eigen-directions exceeding it are signal.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    X = np.asarray(data, dtype=float)
    T, N = X.shape
    if missing_mask is None:
        missing_mask = ~np.isfinite(X)
    observed = ~missing_mask

    def _corr_top(mat: np.ndarray) -> float:
        # standardize columns over observed entries; fill missing with 0 (mean)
        Z = np.zeros_like(mat)
        for j in range(mat.shape[1]):
            col = mat[:, j]
            obs = np.isfinite(col)
            if obs.sum() < 2:
                continue
            mu = col[obs].mean()
            sd = col[obs].std()
            if sd < 1e-12:
                continue
            z = np.zeros_like(col)
            z[obs] = (col[obs] - mu) / sd
            Z[:, j] = z
        C = (Z.T @ Z) / max(T - 1, 1)
        w = np.linalg.eigvalsh(C)
        return float(w[-1])

    null_stats = np.empty(n_perm)
    for k in range(n_perm):
        Xp = np.full_like(X, np.nan)
        for j in range(N):
            obs_idx = np.where(observed[:, j])[0]
            if len(obs_idx) == 0:
                continue
            vals = X[obs_idx, j]
            perm = rng.permutation(vals)
            Xp[obs_idx, j] = perm
        null_stats[k] = _corr_top(Xp)
    floor = float(np.quantile(null_stats, quantile))
    return NoiseFloorResult(
        floor=floor,
        kind="empirical_permutation",
        reason=f"{int(quantile*100)}th pct of permutation null of {statistic}",
        assumptions_hold=True,
        detail={
            "n_perm": n_perm,
            "quantile": quantile,
            "null_mean": float(null_stats.mean()),
            "null_max": float(null_stats.max()),
        },
    )


# --------------------------------------------------------------------------- #
# Floor selection (NF-04)
# --------------------------------------------------------------------------- #
def estimate(
    data: np.ndarray,
    missing_mask: Optional[np.ndarray] = None,
    n_perm: int = 500,
    quantile: float = 0.99,
    rng: Optional[np.random.Generator] = None,
    balanced_tolerance: float = 1e-9,
) -> NoiseFloorResult:
    """Choose analytic vs empirical floor by checking the balanced-panel assumption.

    If the panel is balanced (no missingness), the analytic MP edge applies.
    Otherwise — the usual case under the sparse/asynchronous metrology of §DM-02
    — the empirical floor takes precedence and the reason is recorded (NF-02/04).
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
    res = empirical_floor(X, missing_mask, n_perm=n_perm, quantile=quantile, rng=rng)
    res.assumptions_hold = True
    res.reason += (
        f"; analytic MP refused because panel is unbalanced "
        f"(frac_missing={frac_missing:.3f}, balanced-panel assumption violated, NF-02)"
    )
    res.detail["frac_missing"] = frac_missing
    return res


# --------------------------------------------------------------------------- #
# Minimum-variance / Harris benchmark (NF-01) — the control noise floor
# --------------------------------------------------------------------------- #
def minimum_variance_benchmark(output: np.ndarray, delay: int = 1) -> float:
    """Harris minimum-variance benchmark for one output (NF-01, DIAG-05).

    The minimum achievable output variance under any controller equals the
    variance of the ``delay``-step-ahead forecast error of the output — the part
    no controller can pre-empt because of the process/measurement delay.  We
    estimate it by fitting a short AR model and taking the variance of the
    ``delay``-step-ahead prediction error (the impulse-response-truncated
    innovation variance).  This is the control-theoretic analog of the
    Marchenko-Pastur noise edge.
    """
    y = np.asarray(output, dtype=float)
    y = y[np.isfinite(y)]
    n = len(y)
    if n < 4 * (delay + 2):
        return float(np.var(y)) if n else float("nan")
    p = min(max(2 * delay, 2), n // 4)
    # fit AR(p) by least squares
    Yt = y[p:]
    X = np.column_stack([y[p - k - 1:n - k - 1] for k in range(p)])
    X = np.column_stack([np.ones(len(Yt)), X])
    beta, *_ = np.linalg.lstsq(X, Yt, rcond=None)
    # build psi-weights (impulse response) and sum first `delay` terms
    ar = beta[1:]
    psi = np.zeros(delay)
    psi[0] = 1.0
    for i in range(1, delay):
        psi[i] = sum(ar[k] * psi[i - k - 1] for k in range(min(i, len(ar))))
    resid = Yt - X @ beta
    sigma_a2 = float(np.var(resid))
    return float(sigma_a2 * np.sum(psi ** 2))


def harris_index(output: np.ndarray, delay: int = 1,
                 mv_benchmark: Optional[float] = None) -> float:
    """Harris index = achieved variance / minimum-variance benchmark (NF-01).

    A value near 1 means the controller is already at the achievability floor;
    values well above 1 mean recoverable variance remains (DIAG-05).
    """
    y = np.asarray(output, dtype=float)
    y = y[np.isfinite(y)]
    if mv_benchmark is None:
        mv_benchmark = minimum_variance_benchmark(y, delay)
    achieved = float(np.var(y))
    if mv_benchmark is None or mv_benchmark <= 0:
        return float("nan")
    return achieved / mv_benchmark


# --------------------------------------------------------------------------- #
# Eigenvalue cleaning (NF-05)
# --------------------------------------------------------------------------- #
def eigenvalue_clip(cov: np.ndarray, floor: float) -> np.ndarray:
    """Clip eigenvalues below ``floor`` up to the floor (NF-05; SVD-truncation analog).

    Stabilizes an identification covariance by refusing to trust directions whose
    variance is indistinguishable from noise, without dropping them to zero.
    """
    w, V = np.linalg.eigh(np.asarray(cov, dtype=float))
    w_clipped = np.where(w < floor, floor, w)
    return (V * w_clipped) @ V.T


def ledoit_peche_shrinkage(cov: np.ndarray, q: float, alpha: Optional[float] = None) -> np.ndarray:
    """Nonlinear (Ledoit-Peche-style) shrinkage toward the bulk (NF-05).

    NOTE: nonlinear shrinkage is itself derived under the balanced-panel
    assumption; when panels are unbalanced the caller should prefer
    cross-validated cleaning strength (§VAL) over this analytic form.  ``alpha``
    (if given) is an explicit linear-shrinkage fallback toward the scaled
    identity, which makes the unbalanced-panel escape hatch usable.
    """
    cov = np.asarray(cov, dtype=float)
    w, V = np.linalg.eigh(cov)
    if alpha is not None:
        mu = float(np.mean(w))
        w_shrunk = (1 - alpha) * w + alpha * mu
        return (V * w_shrunk) @ V.T
    # Simple nonlinear shrinkage: pull eigenvalues toward the MP-implied bulk mean
    _, upper = marchenko_pastur_edge(q)
    bulk = w[w <= upper * np.median(w[w > 0]) if np.any(w > 0) else w]
    target = float(np.mean(bulk)) if bulk.size else float(np.mean(w))
    # shrink small eigenvalues more than large ones
    w_pos = np.clip(w, 1e-12, None)
    weight = upper / (upper + w_pos)
    w_shrunk = (1 - weight) * w + weight * target
    return (V * w_shrunk) @ V.T
