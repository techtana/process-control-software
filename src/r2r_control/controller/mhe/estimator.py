"""Moving-horizon estimator (State QR) — SIM-03.

A Kalman-filter realization of the MHE (the limiting case of the moving-horizon
MAP estimate). The estimator weights (State QR) set the process-noise vs
measurement-noise trade-off and are independent of the controller weights.
"""

from __future__ import annotations

import numpy as np


class KalmanMHE:
    """Disturbance/tool-state estimator with configurable State QR."""

    def __init__(self, A_d: np.ndarray, Q: np.ndarray, R: np.ndarray,
                 d0: np.ndarray, P0: np.ndarray):
        self.A_d = np.asarray(A_d, dtype=float)
        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float)
        self.d_hat = np.asarray(d0, dtype=float).copy()
        self.P = np.asarray(P0, dtype=float).copy()

    def predict_state(self) -> np.ndarray:
        """One-step model prediction of the disturbance (no measurement)."""
        return self.A_d @ self.d_hat

    def innovation(self, y, control_contribution, ff_contribution) -> np.ndarray:
        """One-step prediction error: the part the model did not anticipate."""
        return y - (control_contribution + ff_contribution) - self.A_d @ self.d_hat

    def update(self, innov: np.ndarray) -> np.ndarray:
        """Kalman time + measurement update; returns the new state estimate."""
        P_pred = self.A_d @ self.P @ self.A_d.T + self.Q
        K = P_pred @ np.linalg.inv(P_pred + self.R)
        self.d_hat = self.A_d @ self.d_hat + K @ innov
        self.P = (np.eye(len(self.d_hat)) - K) @ P_pred
        return self.d_hat

    def propagate(self) -> np.ndarray:
        """Time update only (no measurement this event)."""
        self.d_hat = self.A_d @ self.d_hat
        self.P = self.A_d @ self.P @ self.A_d.T + self.Q
        return self.d_hat
