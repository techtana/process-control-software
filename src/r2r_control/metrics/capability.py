"""Capability metrics with finite-sample uncertainty (SIM-06, E10).

Cpk, %OOS, mean-off-target, std and Harris per output. On real, sparse,
asynchronous data each carries finite-sample confidence intervals and a
minimum-sample gate suppresses under-powered metrics. This is separate from,
and in addition to, the model-uncertainty Monte Carlo of §SIM-02.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
from scipy.stats import norm

from ..core.noise_floor.analytic import harris_index
from .harris import harris_with_ci
from .min_samples import min_sample_gate, INSUFFICIENT


def cpk_point(mean: float, std: float, lsl: float, usl: float) -> float:
    return min(usl - mean, mean - lsl) / (3.0 * max(std, 1e-12))


def cpk_confidence_interval(cpk: float, n: int, alpha: float = 0.05):
    """Large-sample CI for Cpk: Var(Cpk_hat) ~ 1/(9n) + Cpk^2 / (2(n-1))."""
    if n < 2 or not np.isfinite(cpk):
        return float("nan"), float("nan")
    se = np.sqrt(1.0 / (9.0 * n) + cpk ** 2 / (2.0 * (n - 1)))
    z = norm.ppf(1 - alpha / 2)
    return float(cpk - z * se), float(cpk + z * se)


@dataclass
class Metrics:
    """Per-output capability and control metrics, with uncertainty when available."""

    std: np.ndarray
    mean_off_target: np.ndarray
    cpk: np.ndarray
    pct_oos: np.ndarray
    harris: np.ndarray
    cpk_ci: Optional[np.ndarray] = None        # (n_out, 2) finite-sample CIs
    harris_ci: Optional[np.ndarray] = None     # (n_out, 2)
    suppressed: Optional[np.ndarray] = None    # bool per output (E10 gate)
    n_effective: Optional[np.ndarray] = None
    bands: Dict = field(default_factory=dict)  # model-uncertainty MC bands (SIM-02)

    def to_dict(self):
        d = {"std": self.std.tolist(), "mean_off_target": self.mean_off_target.tolist(),
             "cpk": self.cpk.tolist(), "pct_oos": self.pct_oos.tolist(),
             "harris": self.harris.tolist()}
        if self.cpk_ci is not None:
            d["cpk_ci"] = np.asarray(self.cpk_ci).tolist()
        if self.harris_ci is not None:
            d["harris_ci"] = np.asarray(self.harris_ci).tolist()
        if self.suppressed is not None:
            d["suppressed"] = np.asarray(self.suppressed).tolist()
            d["note"] = [INSUFFICIENT if s else "" for s in self.suppressed]
        if self.bands:
            d["bands"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                          for k, v in self.bands.items()}
        return d


def compute_metrics(y: np.ndarray, targets: np.ndarray, lsl: np.ndarray, usl: np.ndarray,
                    delay: int = 1, n_effective: Optional[np.ndarray] = None,
                    min_samples: Optional[int] = None, finite_sample_ci: bool = False,
                    rng: Optional[np.random.Generator] = None) -> Metrics:
    """Cpk, std, mean-off-target, %OOS, Harris per output (SIM-06).

    With ``finite_sample_ci`` (real sparse data) per-output Cpk/Harris intervals
    are attached and, given ``min_samples``, under-powered outputs are suppressed
    (E10). Without it (abundant simulated output) the gate is inactive.
    """
    y = np.asarray(y, dtype=float)
    n_out = y.shape[1]
    std = np.nanstd(y, axis=0)
    mean = np.nanmean(y, axis=0)
    cpk = np.empty(n_out)
    pct_oos = np.empty(n_out)
    harris = np.empty(n_out)
    cpk_ci = np.full((n_out, 2), np.nan)
    harris_ci = np.full((n_out, 2), np.nan)
    suppressed = np.zeros(n_out, dtype=bool)
    for j in range(n_out):
        col = y[:, j]
        col = col[np.isfinite(col)]
        n_eff = int(n_effective[j]) if n_effective is not None else len(col)
        cpk[j] = cpk_point(mean[j], std[j], lsl[j], usl[j])
        pct_oos[j] = 100.0 * np.mean((col < lsl[j]) | (col > usl[j])) if len(col) else np.nan
        if finite_sample_ci:
            harris[j], harris_ci[j, 0], harris_ci[j, 1] = harris_with_ci(col, delay=delay,
                                                                         rng=rng)
            cpk_ci[j] = cpk_confidence_interval(cpk[j], n_eff)
            if min_samples is not None and min_sample_gate(n_eff, min_samples):
                suppressed[j] = True
                cpk[j] = np.nan
                harris[j] = np.nan
        else:
            harris[j] = harris_index(col, delay=delay)
    return Metrics(std=std, mean_off_target=np.abs(mean - targets), cpk=cpk,
                   pct_oos=pct_oos, harris=harris,
                   cpk_ci=cpk_ci if finite_sample_ci else None,
                   harris_ci=harris_ci if finite_sample_ci else None,
                   suppressed=suppressed if finite_sample_ci else None,
                   n_effective=n_effective)
