"""Feedback (closed-loop endogeneity) test (DQ-04, EX-01; modelling assumption M3).

Collinearity checks (VIF, rank) cannot see the main closed-loop problem: a
controller sets the knobs *in response to* past outputs, so the knob settings
are correlated with the disturbance they are correcting. Regressing outputs on
such knobs gives a biased gain, however well-conditioned the regressors look.

This module runs a Granger-style test per knob: does the history of measured
outputs predict the knob setting beyond the knob's own previous value? Under
open-loop (exogenous) excitation it should not; under feedback it does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy.stats import f as _f_dist


@dataclass
class FeedbackTest:
    detected: bool
    pvalue: Optional[float]                     # Bonferroni-adjusted minimum over knobs
    per_knob_pvalue: Dict[str, float] = field(default_factory=dict)
    n_used: int = 0
    n_lags: int = 2
    detail: str = ""

    def to_dict(self):
        return {"detected": self.detected, "pvalue": self.pvalue,
                "per_knob_pvalue": self.per_knob_pvalue, "n_used": self.n_used,
                "n_lags": self.n_lags, "detail": self.detail}


def _lagged_measured_outputs(Y: np.ndarray, measured: np.ndarray, n_lags: int) -> np.ndarray:
    """Row t holds the ``n_lags`` most recent measured outputs strictly before t."""
    T, n_out = Y.shape
    out = np.full((T, n_lags * n_out), np.nan)
    history: List[int] = []
    for t in range(T):
        if len(history) >= n_lags:
            out[t] = np.concatenate([Y[history[-k]] for k in range(1, n_lags + 1)])
        if measured[t]:
            history.append(t)
    return out


def feedback_test(used_knobs: np.ndarray, y: np.ndarray, measured: Optional[np.ndarray] = None,
                  exclude: Optional[np.ndarray] = None, knob_names: Optional[List[str]] = None,
                  n_lags: int = 2, alpha: float = 0.01) -> FeedbackTest:
    """Granger-style test for knob settings that respond to past outputs.

    For each knob, regress ``u_t`` on ``[1, u_{t-1}, y_lags]`` and F-test the
    ``y_lags`` block, where ``y_lags`` are the ``n_lags`` most recent measured
    output vectors before ``t``. Events in ``exclude`` (e.g. control-off
    episodes) are dropped. Feedback is ``detected`` when the Bonferroni-adjusted
    minimum p-value over knobs is below ``alpha``.
    """
    U = np.asarray(used_knobs, dtype=float)
    Y = np.asarray(y, dtype=float)
    T, n_knob = U.shape
    names = knob_names or [f"u{j}" for j in range(n_knob)]
    if measured is None:
        measured = np.all(np.isfinite(Y), axis=1)
    measured = np.asarray(measured, dtype=bool)
    lags = _lagged_measured_outputs(Y, measured, n_lags)
    ok = np.all(np.isfinite(lags), axis=1)
    ok[0] = False
    if exclude is not None:
        excl = np.asarray(exclude, dtype=bool)
        ok &= ~excl
        ok[1:] &= ~excl[:-1]          # the previous knob value must be a controlled one too
    idx = np.where(ok)[0]
    q = lags.shape[1]
    if len(idx) < q + 10:
        return FeedbackTest(False, None, n_used=len(idx), n_lags=n_lags,
                            detail="too few events with output history to test feedback")

    pvals: Dict[str, float] = {}
    for j in range(n_knob):
        u_t, u_prev = U[idx, j], U[idx - 1, j]
        if np.nanstd(u_t) < 1e-12:
            continue
        X0 = np.column_stack([np.ones(len(idx)), u_prev])
        X1 = np.column_stack([X0, lags[idx]])
        r0 = u_t - X0 @ np.linalg.lstsq(X0, u_t, rcond=None)[0]
        r1 = u_t - X1 @ np.linalg.lstsq(X1, u_t, rcond=None)[0]
        ssr0, ssr1 = float(r0 @ r0), float(r1 @ r1)
        dof = len(idx) - X1.shape[1]
        if ssr1 <= 1e-15 or dof <= 0:
            continue
        F = ((ssr0 - ssr1) / q) / (ssr1 / dof)
        pvals[names[j]] = float(_f_dist.sf(max(F, 0.0), q, dof))
    if not pvals:
        return FeedbackTest(False, None, n_used=len(idx), n_lags=n_lags,
                            detail="no knob varies outside excluded events")
    p_adj = float(min(1.0, min(pvals.values()) * len(pvals)))
    detected = p_adj < alpha
    return FeedbackTest(
        detected=bool(detected), pvalue=p_adj, per_knob_pvalue=pvals, n_used=len(idx),
        n_lags=n_lags,
        detail=("knob settings respond to past outputs (closed-loop feedback): rows are "
                "endogenous and must not be used for gain identification (DQ-06, M3)"
                if detected else "no evidence that knob settings respond to past outputs"))
