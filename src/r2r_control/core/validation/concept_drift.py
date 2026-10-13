"""Concept-drift test: does the relationship itself move? (VAL-06)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ConceptDriftResult:
    slope: float
    degrades_with_horizon: bool
    detail: str

    def to_dict(self):
        return {"slope": float(self.slope), "degrades_with_horizon": self.degrades_with_horizon,
                "detail": self.detail}


def concept_drift_test(horizons: np.ndarray, errors: np.ndarray,
                       slope_tol: float = 1e-9) -> ConceptDriftResult:
    """Test whether forward error degrades systematically with horizon (VAL-06)."""
    h = np.asarray(horizons, dtype=float)
    e = np.asarray(errors, dtype=float)
    good = np.isfinite(h) & np.isfinite(e)
    h, e = h[good], e[good]
    if len(h) < 3 or np.ptp(h) == 0:
        return ConceptDriftResult(0.0, False, "insufficient or non-varying horizons")
    slope = float(np.polyfit(h, e, 1)[0])
    degrades = slope > slope_tol and np.corrcoef(h, e)[0, 1] > 0.5
    return ConceptDriftResult(
        slope=slope, degrades_with_horizon=bool(degrades),
        detail="error rises with horizon => relationship is drifting" if degrades
        else "no systematic horizon degradation")
