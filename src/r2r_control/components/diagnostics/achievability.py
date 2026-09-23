"""DIAG-05 — achievability verdict (the MP / Ledoit-Peche analog for control)."""

from __future__ import annotations

from typing import Dict

import numpy as np

from ...core.noise_floor.analytic import harris_index


def achievability_verdict(y_observed: np.ndarray, delay: int = 1,
                          floor_tolerance: float = 1.25) -> Dict:
    """Is the controller already at the minimum-variance floor? (DIAG-05)."""
    Y = np.asarray(y_observed, dtype=float)
    harris = [harris_index(Y[:, j], delay) for j in range(Y.shape[1])]
    mean_h = float(np.nanmean(harris))
    at_floor = mean_h <= floor_tolerance
    verdict = ("the controller is working as well as it could; the remaining variance is "
               "unexplained / irreducible (at the minimum-variance floor, Harris≈1)"
               if at_floor else
               f"achieved variance sits {mean_h:.2f}x above the minimum-variance floor; a "
               f"recoverable gap exists, route to FF/gain/QR cause")
    return {"harris_per_output": [float(h) for h in harris], "mean_harris": mean_h,
            "at_achievability_floor": bool(at_floor), "verdict": verdict}
