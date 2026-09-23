"""Counterfactual reconstruction by subtraction (§4.5, CF-01)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ...contracts.shapes import check_shape_contract


@dataclass
class CounterfactualResult:
    y_nocontrol: np.ndarray
    control_contribution: np.ndarray
    u0: np.ndarray
    u0_source: str
    u0_uncertainty: Optional[np.ndarray] = None
    bands: Optional[dict] = field(default=None)

    def to_dict(self):
        d = {"u0": np.asarray(self.u0).tolist(), "u0_source": self.u0_source,
             "control_contribution_var": np.nanvar(self.control_contribution, axis=0).tolist()}
        if self.bands is not None:
            d["bands"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                          for k, v in self.bands.items()}
        return d


def reconstruct(y_observed: np.ndarray, M: np.ndarray, u_used: np.ndarray, u0: np.ndarray,
                u0_source: str = "unspecified", n_out: Optional[int] = None,
                n_knob: Optional[int] = None) -> CounterfactualResult:
    """Reconstruct the no-control output by subtraction (CF-01).

    ``u0`` may be a single baseline ``(n_knob,)`` or a time-varying baseline
    ``(T, n_knob)`` (A2: a drifting tool's open-loop point drifts).
    """
    y_observed = np.asarray(y_observed, dtype=float)
    u_used = np.asarray(u_used, dtype=float)
    u0 = np.asarray(u0, dtype=float)
    n_out = y_observed.shape[1] if n_out is None else n_out
    n_knob = u_used.shape[1] if n_knob is None else n_knob
    M = check_shape_contract(M, n_out, n_knob, where="counterfactual M")
    u0_row = u0 if u0.ndim == 2 else u0.reshape(1, -1)
    control_contribution = (u_used - u0_row) @ M.T
    return CounterfactualResult(y_nocontrol=y_observed - control_contribution,
                                control_contribution=control_contribution,
                                u0=u0, u0_source=u0_source)


def realized_gain(y_observed: np.ndarray, u_used: np.ndarray,
                  measured_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Realized gain Δoutput/Δknob estimated from observed data (DIAG-02 support)."""
    y = np.asarray(y_observed, dtype=float)
    u = np.asarray(u_used, dtype=float)
    if measured_mask is not None:
        idx = np.where(measured_mask)[0]
        y, u = y[idx], u[idx]
    dy = np.diff(y, axis=0)
    du = np.diff(u, axis=0)
    good = np.all(np.isfinite(dy), axis=1) & np.all(np.isfinite(du), axis=1)
    dy, du = dy[good], du[good]
    if len(du) < du.shape[1] + 1:
        return np.full((y.shape[1], u.shape[1]), np.nan)
    G_T, *_ = np.linalg.lstsq(du, dy, rcond=None)
    return G_T.T
