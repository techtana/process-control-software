"""Analytic noise floors (NF-01/02): Marchenko-Pastur edge and Harris benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


def marchenko_pastur_edge(q: float) -> Tuple[float, float]:
    """Lower/upper edges of the MP bulk for aspect ratio ``q = N/T`` (NF-01)."""
    sq = np.sqrt(q)
    return (1.0 - sq) ** 2, (1.0 + sq) ** 2


@dataclass
class NoiseFloorResult:
    floor: float
    kind: str
    reason: str
    assumptions_hold: bool
    detail: dict

    def to_dict(self):
        return {"floor": float(self.floor), "kind": self.kind, "reason": self.reason,
                "assumptions_hold": self.assumptions_hold, "detail": self.detail}


def analytic_floor(N: int, T: int, variance: float = 1.0) -> NoiseFloorResult:
    """MP analytic floor, with an explicit balanced-panel assumption (NF-01/02)."""
    q = N / T
    _, upper = marchenko_pastur_edge(q)
    return NoiseFloorResult(
        floor=float(variance * upper), kind="analytic_mp",
        reason="balanced-panel MP edge (1+sqrt(N/T))^2", assumptions_hold=True,
        detail={"q": q, "upper_edge": upper, "variance": variance},
    )


def minimum_variance_benchmark(output: np.ndarray, delay: int = 1) -> float:
    """Harris minimum-variance benchmark for one output (NF-01, DIAG-05).

    Fit an AR model to the output, take the first ``delay`` impulse-response
    coefficients, and scale the innovation variance by their squared sum: the
    variance an ideal minimum-variance controller would leave behind.
    """
    y = np.asarray(output, dtype=float)
    y = y[np.isfinite(y)]
    n = len(y)
    if n < 4 * (delay + 2):
        return float(np.var(y)) if n else float("nan")
    p = min(max(2 * delay, 2), n // 4)
    Yt = y[p:]
    X = np.column_stack([y[p - k - 1:n - k - 1] for k in range(p)])
    X = np.column_stack([np.ones(len(Yt)), X])
    beta, *_ = np.linalg.lstsq(X, Yt, rcond=None)
    ar = beta[1:]
    psi = np.zeros(delay)
    psi[0] = 1.0
    for i in range(1, delay):
        psi[i] = sum(ar[k] * psi[i - k - 1] for k in range(min(i, len(ar))))
    sigma_a2 = float(np.var(Yt - X @ beta))
    return float(sigma_a2 * np.sum(psi ** 2))


def harris_index(output: np.ndarray, delay: int = 1,
                 mv_benchmark: Optional[float] = None) -> float:
    """Harris index = achieved variance / minimum-variance benchmark (NF-01)."""
    y = np.asarray(output, dtype=float)
    y = y[np.isfinite(y)]
    if mv_benchmark is None:
        mv_benchmark = minimum_variance_benchmark(y, delay)
    if mv_benchmark is None or not mv_benchmark > 0:
        return float("nan")
    return float(np.var(y)) / mv_benchmark
