"""DIAG-02 — gain-mismatch instability diagnostic, plus the A1/E1 drift monitor."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ...core.counterfactual.reconstruct import realized_gain


def gain_mismatch(y_observed: np.ndarray, u_used: np.ndarray, M_model: np.ndarray,
                  innovation: Optional[np.ndarray] = None,
                  measured_mask: Optional[np.ndarray] = None) -> Dict:
    """Detect over-correction from an underestimated gain (DIAG-02).

    Compares realized gain (Δoutput/Δknob via the counterfactual service) with
    the model gain, and flags oscillation in the output level (negative lag-1
    autocorrelation = ringing). Separates "moves too large because the gain is
    underestimated" from genuine disturbance growth.
    """
    G_real = realized_gain(y_observed, u_used, measured_mask)
    M_model = np.asarray(M_model, dtype=float)
    ratio = np.full_like(M_model, np.nan)
    mask = np.abs(M_model) > 1e-9
    ratio[mask] = G_real[mask] / M_model[mask]
    median_ratio = float(np.nanmedian(ratio))
    underestimated = median_ratio > 1.3

    Y = np.asarray(y_observed, dtype=float)
    ac1 = []
    for j in range(Y.shape[1]):
        col = Y[:, j]
        col = col[np.isfinite(col)]
        col = col - col.mean()
        if len(col) > 6 and np.std(col) > 1e-9:
            ac1.append(np.corrcoef(col[1:], col[:-1])[0, 1])
    mean_ac1 = float(np.mean(ac1)) if ac1 else np.nan
    oscillating = bool(np.isfinite(mean_ac1) and mean_ac1 < -0.3)

    growing = False
    if innovation is not None:
        iv = np.asarray(innovation, dtype=float)
        half = len(iv) // 2
        if half > 5:
            growing = bool(np.nanvar(iv[half:]) > 2.0 * np.nanvar(iv[:half]))
    if oscillating and underestimated:
        cause = "gain_underestimated_overcorrection"
    elif oscillating and growing:
        cause = "disturbance_growth"
    elif oscillating:
        cause = "oscillation_unattributed"
    else:
        cause = "none"
    return {"realized_vs_model_ratio_median": median_ratio,
            "gain_underestimated": bool(underestimated),
            "oscillation_lag1_autocorr": mean_ac1, "oscillating": oscillating,
            "disturbance_growth": growing, "diagnosis": cause}


def gain_drift_monitor(y_observed: np.ndarray, u_used: np.ndarray, window: int = 60,
                       measured_mask: Optional[np.ndarray] = None,
                       rel_trend_threshold: float = 0.2) -> Dict:
    """Monitor realized gain over time for drift (A1/E1).

    The architecture assumes a fixed gain with additive disturbance drift. If
    the gain itself drifts, the DOE gain, the counterfactual and the MPC model
    fail together. We estimate the realized gain in half-overlapping windows and
    regress its magnitude on time; a strong, consistent trend is the signature
    of gain drift and triggers re-identification.
    """
    Y = np.asarray(y_observed, dtype=float)
    U = np.asarray(u_used, dtype=float)
    if measured_mask is not None:
        idx = np.where(measured_mask)[0]
        Y, U = Y[idx], U[idx]
    n = len(Y)
    if n < 3 * window:
        return {"drift_detected": False, "detail": "record too short to monitor",
                "n_windows": 0}
    centers, mags = [], []
    for start in range(0, n - window + 1, max(window // 2, 1)):
        G = realized_gain(Y[start:start + window], U[start:start + window])
        if np.all(np.isfinite(G)):
            mags.append(float(np.linalg.norm(G)))
            centers.append(start + window / 2)
    if len(mags) < 3:
        return {"drift_detected": False, "detail": "too few estimable windows",
                "n_windows": len(mags)}
    c, m = np.asarray(centers), np.asarray(mags)
    slope = float(np.polyfit(c, m, 1)[0])
    rel_trend = slope * float(c.max() - c.min()) / (float(np.mean(m)) + 1e-12)
    corr = float(np.corrcoef(c, m)[0, 1])
    drift = abs(rel_trend) > rel_trend_threshold and abs(corr) > 0.6
    return {"drift_detected": bool(drift), "relative_trend": rel_trend, "trend_corr": corr,
            "realized_gain_magnitudes": mags, "n_windows": len(mags),
            "detail": ("realized gain trends over time => gain drift, re-identify (A1/E1)"
                       if drift else "no systematic realized-gain trend")}
