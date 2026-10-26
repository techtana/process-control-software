"""Model-predictive controller (Controller QR) — SIM-03/04.

A finite-horizon MPC on the static R2R gain with a predicted disturbance
trajectory. Decision variables are the knob moves ``Δu_0 … Δu_{H-1}``; the cost is

    Σ_k  mpc_Q·||M (u_k − u0) + d̂_{t+k} + ff_k − target||²  +  mpc_R·||Δu_k||²

with ``u_k = u_prev + Σ_{i≤k} Δu_i`` and ``d̂_{t+k} = A^k d̂_t`` from the
estimator's disturbance model, so the horizon matters whenever the disturbance
is predicted to evolve. Move limits ``|Δu| ≤ move_limit`` are enforced exactly
(bounded least squares), and optional absolute knob bounds as linear
constraints, instead of clipping an unconstrained solution (KI-4). Only the
first move is applied (receding horizon). The controller weights are
independent of the estimator weights.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import lsq_linear, minimize

from .limits import apply_move_limit


class MPCController:
    """Receding-horizon MPC with exact move limits and optional absolute bounds."""

    def __init__(self, M_model: np.ndarray, u0: np.ndarray, mpc_Q: float = 1.0,
                 mpc_R: float = 0.1, horizon: int = 1, move_limit: Optional[float] = None,
                 A_d: Optional[np.ndarray] = None, u_lower: Optional[Sequence[float]] = None,
                 u_upper: Optional[Sequence[float]] = None):
        G = np.asarray(M_model, dtype=float)
        self.n_out, self.n_knob = G.shape
        self.H = max(int(horizon), 1)
        self.u0 = np.asarray(u0, dtype=float)
        self.move_limit = (None if move_limit is None or not np.isfinite(move_limit)
                           else float(move_limit))
        self.u_lower = None if u_lower is None else np.asarray(u_lower, dtype=float)
        self.u_upper = None if u_upper is None else np.asarray(u_upper, dtype=float)
        A = np.eye(self.n_out) if A_d is None else np.asarray(A_d, dtype=float)
        self.A_pows = [np.linalg.matrix_power(A, k) for k in range(self.H)]

        H, nk = self.H, self.n_knob
        self.L = np.kron(np.tril(np.ones((H, H))), np.eye(nk))          # Δ -> cumulative
        GH = np.kron(np.eye(H), G)
        self.GH_ones = GH @ np.kron(np.ones((H, 1)), np.eye(nk))         # (u_prev - u0) -> y
        self.sq_Q, self.sq_R = np.sqrt(mpc_Q), np.sqrt(mpc_R)
        self.A_ls = np.vstack([self.sq_Q * GH @ self.L, self.sq_R * np.eye(H * nk)])
        self._chol = cho_factor(self.A_ls.T @ self.A_ls + 1e-12 * np.eye(H * nk))

    def disturbance_trajectory(self, d_hat: np.ndarray, bias: Optional[np.ndarray] = None
                               ) -> np.ndarray:
        """``d̂_{t+k} = A^k d̂_t (+ bias)`` for k = 0..H-1."""
        b = 0.0 if bias is None else np.asarray(bias, dtype=float)
        return np.stack([Ak @ d_hat + b for Ak in self.A_pows])

    def _feasible(self, delta: np.ndarray, u_prev: np.ndarray) -> bool:
        tol = 1e-9
        if self.move_limit is not None and np.any(np.abs(delta) > self.move_limit + tol):
            return False
        if self.u_lower is not None or self.u_upper is not None:
            u = u_prev[None, :] + (self.L @ delta).reshape(self.H, self.n_knob)
            if self.u_lower is not None and np.any(u < self.u_lower - tol):
                return False
            if self.u_upper is not None and np.any(u > self.u_upper + tol):
                return False
        return True

    def move(self, d_traj: np.ndarray, target: np.ndarray, u_prev: np.ndarray,
             ff_now: Optional[np.ndarray] = None) -> np.ndarray:
        """Next knob vector (first move of the constrained horizon optimum)."""
        u_prev = np.asarray(u_prev, dtype=float)
        w = np.asarray(d_traj, dtype=float) - np.asarray(target, dtype=float)[None, :]
        if ff_now is not None:
            w = w.copy()
            w[0] = w[0] + ff_now          # future FF unknown (zero-mean); current one known
        c = self.GH_ones @ (u_prev - self.u0) + w.reshape(-1)
        b = np.concatenate([-self.sq_Q * c, np.zeros(self.H * self.n_knob)])
        delta = cho_solve(self._chol, self.A_ls.T @ b)
        if not self._feasible(delta, u_prev):
            delta = self._constrained(b, u_prev, delta)
        return u_prev + delta[:self.n_knob]

    def _constrained(self, b: np.ndarray, u_prev: np.ndarray, x0: np.ndarray) -> np.ndarray:
        n = self.H * self.n_knob
        ml = np.inf if self.move_limit is None else self.move_limit
        if self.u_lower is None and self.u_upper is None:
            return lsq_linear(self.A_ls, b, bounds=(-ml, ml), method="bvls").x
        cons = []
        Lmat = self.L
        base = np.tile(u_prev, self.H)
        if self.u_upper is not None:
            ub = np.tile(self.u_upper, self.H)
            cons.append({"type": "ineq", "fun": lambda x: ub - (base + Lmat @ x),
                         "jac": lambda x: -Lmat})
        if self.u_lower is not None:
            lb = np.tile(self.u_lower, self.H)
            cons.append({"type": "ineq", "fun": lambda x: (base + Lmat @ x) - lb,
                         "jac": lambda x: Lmat})
        A, AtA, Atb = self.A_ls, self.A_ls.T @ self.A_ls, self.A_ls.T @ b
        res = minimize(lambda x: 0.5 * float(np.sum((A @ x - b) ** 2)), np.clip(x0, -ml, ml),
                       jac=lambda x: AtA @ x - Atb, method="SLSQP",
                       bounds=[(-ml, ml)] * n, constraints=cons,
                       options={"maxiter": 200, "ftol": 1e-10})
        x = res.x
        # guarantee feasibility of the applied move even if the solver stopped early
        u1 = apply_move_limit(u_prev + x[:self.n_knob], u_prev, ml)
        if self.u_lower is not None:
            u1 = np.maximum(u1, self.u_lower)
        if self.u_upper is not None:
            u1 = np.minimum(u1, self.u_upper)
        x = x.copy()
        x[:self.n_knob] = u1 - u_prev
        return x


def mpc_move(M_model: np.ndarray, desired: np.ndarray, u_prev: np.ndarray, u0: np.ndarray,
             mpc_Q: float, mpc_R: float, move_limit: float) -> np.ndarray:
    """One-step move: argmin mpc_Q*||M(u-u0) - desired||^2 + mpc_R*||u - u_prev||^2
    subject to ``|u - u_prev| <= move_limit`` (exact, not clipped)."""
    ctrl = MPCController(M_model, u0, mpc_Q, mpc_R, horizon=1, move_limit=move_limit)
    n_out = np.asarray(M_model).shape[0]
    return ctrl.move(np.zeros((1, n_out)), np.asarray(desired, dtype=float), u_prev)
