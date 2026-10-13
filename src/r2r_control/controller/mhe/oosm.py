"""Out-of-sequence / over-horizon measurement handling (A4/E7).

Delayed measurements are assumed to arrive within the moving horizon so the
estimator can fold them in by re-optimizing. When the delay *exceeds* the horizon
(common in R2R metrology) a standard MHE cannot use the late measurement at all.
The guard routes over-horizon measurements to a slow bias / model-correction
estimator instead of the state update, so the information is still used without
corrupting the live state.
"""

from __future__ import annotations

import numpy as np


class SlowBiasEstimator:
    """Folds over-horizon-late measurement residuals into a slow additive bias."""

    def __init__(self, n_out: int, gain: float = 0.1):
        self.bias = np.zeros(n_out)
        self.gain = gain
        self.n_late = 0

    def correction(self) -> np.ndarray:
        return self.bias

    def incorporate_late(self, residual: np.ndarray) -> np.ndarray:
        self.bias = (1 - self.gain) * self.bias + self.gain * np.asarray(residual, dtype=float)
        self.n_late += 1
        return self.bias


def route_measurement(delay: int, horizon: int) -> str:
    """``"state_update"`` if the delay fits in the MHE horizon, else ``"slow_bias"``."""
    return "state_update" if delay <= horizon else "slow_bias"
