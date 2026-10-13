"""Model-predictive controller (Controller QR) — SIM-03.

A finite-horizon LQ first move: with a static gain over the horizon, the optimum
trades tracking error against move suppression. The controller weights
(Controller QR) are independent of the estimator weights.
"""

from __future__ import annotations

import numpy as np

from .limits import apply_move_limit


def mpc_move(M_model: np.ndarray, desired: np.ndarray, u_prev: np.ndarray, u0: np.ndarray,
             mpc_Q: float, mpc_R: float, move_limit: float) -> np.ndarray:
    """Next knob vector: argmin mpc_Q*||M(u-u0) - desired||^2 + mpc_R*||u - u_prev||^2,
    then move-limited."""
    G = np.asarray(M_model, dtype=float)
    H = mpc_Q * (G.T @ G) + mpc_R * np.eye(G.shape[1])
    rhs = mpc_Q * (G.T @ desired) + mpc_R * (u_prev - u0)
    return apply_move_limit(u0 + np.linalg.solve(H, rhs), u_prev, move_limit)
