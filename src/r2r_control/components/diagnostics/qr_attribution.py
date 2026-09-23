"""DIAG-03 — Estimator-QR vs Controller-QR attribution."""

from __future__ import annotations

from typing import Dict

import numpy as np

from ...core import stats


def qr_attribution(innovation: np.ndarray, u_used: np.ndarray, y_observed: np.ndarray,
                   alpha: float = 0.05) -> Dict:
    """Attribute a problem to Controller QR vs State QR (DIAG-03).

    Controller-weight problems show in the move/tracking trade-off;
    estimator-weight problems show in the whiteness of the innovation. Honors the
    FB-08 limit that passive data cannot fully separate gain error from tuning.
    """
    innov = np.asarray(innovation, dtype=float)
    innov2d = innov if innov.ndim > 1 else innov[:, None]
    innov2d = innov2d[np.all(np.isfinite(innov2d), axis=1)]
    lb_p = []
    for j in range(innov2d.shape[1]):
        lb = stats.ljung_box(innov2d[:, j], lags=min(10, len(innov2d) // 3))
        if lb.pvalue is not None:
            lb_p.append(lb.pvalue)
    innov_colored = bool(lb_p and np.nanmin(lb_p) < alpha)

    du = np.diff(np.asarray(u_used, dtype=float), axis=0)
    move_energy = float(np.mean(np.sum(du ** 2, axis=1)))
    Y = np.asarray(y_observed, dtype=float)
    track_energy = float(np.nanmean(np.nansum((Y - np.nanmean(Y, 0)) ** 2, axis=1)))
    move_to_track = move_energy / (track_energy + 1e-9)

    if innov_colored:
        primary = "State_QR"
    elif move_to_track > 1.0:
        primary = "Controller_QR"
    else:
        primary = "neither_dominant"
    return {"innovation_colored": innov_colored,
            "min_ljung_box_p": float(np.nanmin(lb_p)) if lb_p else None,
            "move_to_track_ratio": move_to_track, "primary_suspect": primary,
            "fb08_caveat": ("passive data cannot fully separate gain error from estimator "
                            "tuning; resolution requires excitation or the estimator spec "
                            "(FB-08)")}
