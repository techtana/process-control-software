"""Counterfactual reconstruction service (§4.5, CF-01..04).

Reconstruct what output variation would have looked like *without R2R control
active* by **subtraction, not simulation**.  Because both FF and FB act through
the observed ``u_used``, the control contribution is ``M @ (u_used - u0)``, and
the process sensitivity and per-event disturbance realizations cancel
algebraically:

    y_observed   = disturbance + M @ (u_used - u0)
    y_nocontrol  = y_observed - M @ (u_used - u0)   # == disturbance

This single service is shared: the simulator's no-control arm (§SIM-01), the
regression machine's controller decoupling (§REG-02), and the gain/instability
diagnostics (§DIAG-02) all call it, so "the counterfactual" means exactly one
thing system-wide (§IF-06).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..data.schema import check_shape_contract


@dataclass
class CounterfactualResult:
    y_nocontrol: np.ndarray                # reconstructed open-loop-equivalent output
    control_contribution: np.ndarray       # M @ (u_used - u0), per event
    u0: np.ndarray
    u0_source: str
    u0_uncertainty: Optional[np.ndarray] = None
    bands: Optional[dict] = field(default=None)   # MC uncertainty bands if requested

    def to_dict(self):
        d = {
            "u0": self.u0.tolist(),
            "u0_source": self.u0_source,
            "control_contribution_var": np.nanvar(self.control_contribution, axis=0).tolist(),
        }
        if self.bands is not None:
            d["bands"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                          for k, v in self.bands.items()}
        return d


def reconstruct(
    y_observed: np.ndarray,
    M: np.ndarray,
    u_used: np.ndarray,
    u0: np.ndarray,
    u0_source: str = "unspecified",
    n_out: Optional[int] = None,
    n_knob: Optional[int] = None,
) -> CounterfactualResult:
    """Reconstruct the no-control output by subtraction (CF-01).

    Parameters
    ----------
    y_observed : (T, n_out)
    M : (n_out, n_knob)   process gain (shape contract enforced, DM-04)
    u_used : (T, n_knob)
    u0 : (n_knob,)        the open-loop baseline; the whole counterfactual is
                          sensitive to this being the *true* baseline (CF-02).
    """
    y_observed = np.asarray(y_observed, dtype=float)
    u_used = np.asarray(u_used, dtype=float)
    u0 = np.asarray(u0, dtype=float).reshape(-1)
    if n_out is None:
        n_out = y_observed.shape[1]
    if n_knob is None:
        n_knob = u_used.shape[1]
    M = check_shape_contract(M, n_out, n_knob, where="counterfactual M")
    delta_u = u_used - u0[None, :]
    control_contribution = delta_u @ M.T          # (T, n_out)
    y_nocontrol = y_observed - control_contribution
    return CounterfactualResult(
        y_nocontrol=y_nocontrol,
        control_contribution=control_contribution,
        u0=u0,
        u0_source=u0_source,
    )


def reconstruct_with_bands(
    y_observed: np.ndarray,
    M: np.ndarray,
    u_used: np.ndarray,
    u0: np.ndarray,
    rel_uncertainty_M: float = 0.0,
    u0_uncertainty: Optional[np.ndarray] = None,
    n_samples: int = 200,
    rng: Optional[np.random.Generator] = None,
    quantiles=(0.05, 0.5, 0.95),
    u0_source: str = "unspecified",
) -> CounterfactualResult:
    """Monte-Carlo uncertainty bands on the counterfactual (CF-03, CF-02).

    Accepts a relative-uncertainty estimate on ``M`` (and optional uncertainty on
    the baseline ``u0``) and propagates both via Monte-Carlo to produce bands on
    the reconstructed no-control output.  The baseline sensitivity is surfaced
    explicitly because the entire counterfactual depends on ``u0`` (CF-02).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    y_observed = np.asarray(y_observed, dtype=float)
    u_used = np.asarray(u_used, dtype=float)
    u0 = np.asarray(u0, dtype=float).reshape(-1)
    n_out, n_knob = M.shape
    check_shape_contract(M, n_out, n_knob, where="counterfactual M")

    base = reconstruct(y_observed, M, u_used, u0, u0_source, n_out, n_knob)
    if rel_uncertainty_M <= 0 and u0_uncertainty is None:
        return base

    samples = np.empty((n_samples, *base.y_nocontrol.shape))
    for s in range(n_samples):
        M_s = M * (1.0 + rel_uncertainty_M * rng.standard_normal(M.shape))
        if u0_uncertainty is not None:
            u0_s = u0 + np.asarray(u0_uncertainty, dtype=float) * rng.standard_normal(u0.shape)
        else:
            u0_s = u0
        delta = u_used - u0_s[None, :]
        samples[s] = y_observed - delta @ M_s.T
    qs = np.quantile(samples, quantiles, axis=0)
    base.bands = {
        "quantiles": np.asarray(quantiles),
        "lower": qs[0],
        "median": qs[1] if len(quantiles) > 2 else qs[0],
        "upper": qs[-1],
        "std": samples.std(axis=0),
        "rel_uncertainty_M": rel_uncertainty_M,
    }
    base.u0_uncertainty = u0_uncertainty
    return base


def realized_gain(
    y_observed: np.ndarray,
    u_used: np.ndarray,
    measured_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Estimate realized gain Δoutput/Δknob from observed data (DIAG-02 support).

    A least-squares fit of output deltas on knob deltas between consecutive
    measured events.  Used by the gain-mismatch diagnostic to compare realized
    against model gain.
    """
    y = np.asarray(y_observed, dtype=float)
    u = np.asarray(u_used, dtype=float)
    if measured_mask is not None:
        idx = np.where(measured_mask)[0]
        y = y[idx]
        u = u[idx]
    dy = np.diff(y, axis=0)
    du = np.diff(u, axis=0)
    good = np.all(np.isfinite(dy), axis=1) & np.all(np.isfinite(du), axis=1)
    dy = dy[good]
    du = du[good]
    if len(du) < du.shape[1] + 1:
        return np.full((y.shape[1], u.shape[1]), np.nan)
    # solve dy = du @ G^T  => G = (du^+ dy)^T
    G_T, *_ = np.linalg.lstsq(du, dy, rcond=None)
    return G_T.T
