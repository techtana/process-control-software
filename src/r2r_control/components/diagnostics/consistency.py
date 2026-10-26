"""§13.5 — model-consistency diagnostic using the controller's own prediction ``y_pred``.

The plain realized-gain check conflates an inaccurate ``M`` with FB tracking
lag. The controller's recorded prediction separates them. Let
``e_t = y_t − y_pred_t`` be its prediction error and
``est_t = y_pred_t − M_model (u_t − ū)`` its disturbance estimate (the intercept
absorbs the unknown baseline). Because ``u_t`` responds to ``est_t``, the error
is tested for knob dependence *after* controlling for ``est_t`` and ``e_{t-1}``:

* **M_inaccurate**: the knob settings explain ``e_t`` (``e = δM (u − ū) + …``),
  also after the measured FF inputs are controlled for.
* **M_or_FF_model_error**: the knobs explain ``e_t`` only until the FF inputs
  are controlled for, and the FF inputs explain it. The knob moves are then
  driven by FF compensation, so a gain error and an FF-model error cannot be
  told apart from this data; excitation independent of FF is required.
* **FF_model_error**: the FF inputs explain ``e_t``; the knobs do not.
* **fb_tracking_lag**: neither, but ``e_t`` is strongly autocorrelated: the
  disturbance estimate trails the disturbance (State QR / estimator gain).
* **consistent**: none of the above.

Calibration (synthetic loops, 10 seeds per case, alpha=1e-3): healthy and
well-tuned loops → consistent 10/10; lagging estimator → fb_tracking_lag 10/10;
gain errors of ×2, ×1.3 and ×0.6 → M_inaccurate 10/10 each; FF leakage →
M_or_FF_model_error 10/10. Mean lag-1 autocorrelation of ``e`` is about
0.17–0.33 for the default estimator and 0.52–0.66 for a lagging one.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy.stats import f as _f_dist

UNIDENTIFIABLE_M_FF = "M_or_FF_model_error"


def _block_pvalue(e: np.ndarray, base: np.ndarray, block: np.ndarray) -> float:
    """F-test p-value for adding ``block`` to the regression of ``e`` on ``base``."""
    X1 = np.column_stack([base, block])
    r0 = e - base @ np.linalg.lstsq(base, e, rcond=None)[0]
    r1 = e - X1 @ np.linalg.lstsq(X1, e, rcond=None)[0]
    q, dof = block.shape[1], len(e) - X1.shape[1]
    ssr1 = float(r1 @ r1)
    if q == 0 or dof <= 0 or ssr1 <= 1e-15:
        return 1.0
    F = ((float(r0 @ r0) - ssr1) / q) / (ssr1 / dof)
    return float(_f_dist.sf(max(F, 0.0), q, dof))


def model_consistency(y_observed: np.ndarray, y_pred: np.ndarray, u_used: np.ndarray,
                      M_model: np.ndarray, ff_inputs: Optional[np.ndarray] = None,
                      measured_mask: Optional[np.ndarray] = None,
                      exclude: Optional[np.ndarray] = None, alpha: float = 1e-3,
                      lag_ac1: float = 0.4) -> Dict:
    """Disambiguate inaccurate ``M`` from FB tracking lag (and FF error) via ``y_pred``."""
    Y = np.asarray(y_observed, dtype=float)
    P = np.asarray(y_pred, dtype=float)
    U = np.asarray(u_used, dtype=float)
    M = np.asarray(M_model, dtype=float)
    E = Y - P
    meas = np.all(np.isfinite(E), axis=1)
    if measured_mask is not None:
        mm = np.asarray(measured_mask, dtype=bool)
        meas &= mm if mm.ndim == 1 else np.all(mm, axis=1)
    if exclude is not None:
        meas &= ~np.asarray(exclude, dtype=bool)
    FF = None
    if ff_inputs is not None and np.asarray(ff_inputs).size:
        FF = np.asarray(ff_inputs, dtype=float)
        meas &= np.all(np.isfinite(FF), axis=1)
        if np.all(FF[meas].std(0) < 1e-12):
            FF = None
    prev = np.full_like(E, np.nan)
    last = None
    for t in range(len(E)):
        if last is not None:
            prev[t] = E[last]
        if meas[t]:
            last = t
    rows = np.where(meas & np.all(np.isfinite(prev), axis=1))[0]
    n_out, n_knob = E.shape[1], U.shape[1]
    n_ff = 0 if FF is None else FF.shape[1]
    if len(rows) < n_knob + n_out + n_ff + 10:
        return {"verdict": "insufficient data", "p_knobs": None, "p_knobs_given_ff": None,
                "p_ff_given_knobs": None, "ac1": None, "gain_error_estimate": None,
                "detail": "too few measured events with a previous prediction error"}
    ub = U - U[rows].mean(0)
    est = P - ub @ M.T
    p_u, p_uf, p_fu = [], [], []
    dM = np.zeros((n_out, n_knob))
    for j in range(n_out):
        e = E[rows, j]
        base = np.column_stack([np.ones(len(rows)), est[rows], prev[rows, j]])
        p_u.append(_block_pvalue(e, base, ub[rows]))
        X = np.column_stack([base, ub[rows]] + ([FF[rows]] if FF is not None else []))
        coef = np.linalg.lstsq(X, e, rcond=None)[0]
        dM[j] = coef[base.shape[1]:base.shape[1] + n_knob]
        if FF is not None:
            p_uf.append(_block_pvalue(e, np.column_stack([base, FF[rows]]), ub[rows]))
            p_fu.append(_block_pvalue(e, np.column_stack([base, ub[rows]]), FF[rows]))

    def adj(ps):
        return float(min(1.0, min(ps) * len(ps))) if ps else None

    pu, puf, pfu = adj(p_u), adj(p_uf), adj(p_fu)
    ac1s = []
    for j in range(n_out):
        ec = E[meas, j] - E[meas, j].mean()
        if len(ec) > 3 and np.std(ec) > 1e-12:
            ac1s.append(float(np.corrcoef(ec[1:], ec[:-1])[0, 1]))
    ac1 = float(np.mean(ac1s)) if ac1s else 0.0

    if puf is not None and puf < alpha:
        verdict, detail = "M_inaccurate", ("prediction error depends on the knob settings "
                                           "beyond the FF inputs => the gain model is wrong")
    elif pu < alpha and pfu is not None and pfu < alpha:
        verdict, detail = UNIDENTIFIABLE_M_FF, (
            "knob dependence vanishes once FF inputs are controlled for: the knob moves are "
            "FF-driven, so gain error and FF-model error cannot be separated from this data "
            "(excitation independent of FF required)")
    elif pu < alpha:
        verdict, detail = "M_inaccurate", ("prediction error depends on the knob settings "
                                           "=> the gain model is wrong (not tracking lag)")
    elif pfu is not None and pfu < alpha:
        verdict, detail = "FF_model_error", ("FF inputs explain the prediction error and the "
                                             "knobs do not => the FF model is wrong")
    elif ac1 > lag_ac1:
        verdict, detail = "fb_tracking_lag", (
            "prediction error is strongly autocorrelated without knob or FF dependence => the "
            "disturbance estimate lags (State QR / estimator gain too low)")
    else:
        verdict, detail = "consistent", ("no knob or FF dependence and no strong lag in the "
                                         "controller's prediction error")
    return {"verdict": verdict, "p_knobs": pu, "p_knobs_given_ff": puf,
            "p_ff_given_knobs": pfu, "ac1": ac1, "gain_error_estimate": dM.tolist(),
            "n_used": int(len(rows)), "detail": detail}
