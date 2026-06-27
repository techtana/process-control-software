"""Component 3 — MHE / MPC Controller Simulator (§7, SIM-01..07).

Simulate process outcomes **with and without** control in place, reporting
capability and control metrics, while accepting and propagating the errors /
unexplained variance in both the FB and FF models.  The no-control arm is the
algebraic counterfactual from the shared service (§CF), not an independently
coded simulation.  Fully configurable and deterministic given a seed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..foundations import noise_floor as nf
from ..foundations.counterfactual import reconstruct


# --------------------------------------------------------------------------- #
# Configuration (SIM-03/04/05)
# --------------------------------------------------------------------------- #
@dataclass
class PlantConfig:
    """The disturbance/drift model driving the plant (SIM-05)."""

    A_d: Optional[np.ndarray] = None        # disturbance dynamics (n_out x n_out)
    drift_std: float = 0.01                 # random-walk drift innovation
    ar_coef: float = 0.85                   # autocorrelated disturbance
    process_noise_std: float = 0.3
    meas_noise_std: float = 0.2             # measurement-tool uncertainty (SIM-04)
    ff_disturbance_std: float = 0.5         # measurable disturbance amplitude


@dataclass
class ControllerConfig:
    """MHE (State QR) and MPC (Controller QR) configuration (SIM-03/04).

    The two weight sets are independently addressable (SIM-03).
    """

    # MHE / State QR
    mhe_Q: float = 1.0                       # process-noise weight (trust model)
    mhe_R: float = 1.0                       # measurement-noise weight (trust meas)
    mhe_horizon: int = 1                     # moving-horizon window length
    # MPC / Controller QR
    mpc_Q: float = 1.0                       # tracking-error weight
    mpc_R: float = 0.1                       # move-suppression weight
    mpc_horizon: int = 5
    move_limit: float = 0.5                  # max |Δu| per event (rate limit, SIM-04)
    sampling_rate: int = 1                   # measure every k-th event (SIM-04)
    metrology_delay: int = 0                 # asynchronous/delayed metrology (SIM-04)


@dataclass
class Metrics:
    """Per-output capability & control metrics with uncertainty bands (SIM-06)."""

    std: np.ndarray
    mean_off_target: np.ndarray
    cpk: np.ndarray
    pct_oos: np.ndarray
    harris: np.ndarray
    bands: Dict = field(default_factory=dict)

    def to_dict(self):
        d = {
            "std": self.std.tolist(),
            "mean_off_target": self.mean_off_target.tolist(),
            "cpk": self.cpk.tolist(),
            "pct_oos": self.pct_oos.tolist(),
            "harris": self.harris.tolist(),
        }
        if self.bands:
            d["bands"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                          for k, v in self.bands.items()}
        return d


@dataclass
class SimulationResult:
    y_on: np.ndarray
    y_off: np.ndarray
    u: np.ndarray
    state_est: np.ndarray
    innovation: np.ndarray
    disturbance: np.ndarray
    ff_inputs: np.ndarray
    metrics_on: Metrics
    metrics_off: Metrics
    config: Dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "metrics_on": self.metrics_on.to_dict(),
            "metrics_off": self.metrics_off.to_dict(),
            "config": self.config,
        }


def compute_metrics(
    y: np.ndarray,
    targets: np.ndarray,
    lsl: np.ndarray,
    usl: np.ndarray,
    delay: int = 1,
) -> Metrics:
    """Cpk, std, mean-off-target, %OOS, Harris per output (SIM-06)."""
    y = np.asarray(y, dtype=float)
    n_out = y.shape[1]
    std = np.nanstd(y, axis=0)
    mean = np.nanmean(y, axis=0)
    mot = np.abs(mean - targets)
    cpk = np.empty(n_out)
    pct_oos = np.empty(n_out)
    harris = np.empty(n_out)
    for j in range(n_out):
        s = std[j] if std[j] > 1e-12 else 1e-12
        cpk[j] = min(usl[j] - mean[j], mean[j] - lsl[j]) / (3.0 * s)
        col = y[:, j]
        col = col[np.isfinite(col)]
        pct_oos[j] = 100.0 * np.mean((col < lsl[j]) | (col > usl[j])) if len(col) else np.nan
        harris[j] = nf.harris_index(col, delay=delay)
    return Metrics(std=std, mean_off_target=mot, cpk=cpk, pct_oos=pct_oos, harris=harris)


class Simulator:
    """Component 3.  Configurable MHE/MPC closed-loop simulator."""

    def __init__(
        self,
        M: np.ndarray,
        targets: np.ndarray,
        lsl: np.ndarray,
        usl: np.ndarray,
        u0: Optional[np.ndarray] = None,
        FF_gain: Optional[np.ndarray] = None,
        plant: Optional[PlantConfig] = None,
        controller: Optional[ControllerConfig] = None,
        rel_uncertainty_M: float = 0.0,
        rel_uncertainty_FF: float = 0.0,
        model_gain_scale: float = 1.0,
        seed: int = 0,
    ):
        self.M = np.asarray(M, dtype=float)
        self.n_out, self.n_knob = self.M.shape
        # The controller's *model* gain = plant gain * model_gain_scale.  A scale
        # below 1 means the model underestimates the gain, so the controller
        # over-corrects and oscillates — the gain-mismatch instability of DIAG-02.
        self.model_gain_scale = model_gain_scale
        self.targets = np.asarray(targets, dtype=float)
        self.lsl = np.asarray(lsl, dtype=float)
        self.usl = np.asarray(usl, dtype=float)
        self.u0 = np.zeros(self.n_knob) if u0 is None else np.asarray(u0, dtype=float)
        self.FF_gain = None if FF_gain is None else np.asarray(FF_gain, dtype=float)
        self.plant = plant or PlantConfig()
        self.controller = controller or ControllerConfig()
        self.rel_uncertainty_M = rel_uncertainty_M
        self.rel_uncertainty_FF = rel_uncertainty_FF
        self.seed = seed

    # ------------------------------------------------------------------ #
    def _run_once(self, n_steps: int, rng: np.random.Generator,
                  M_plant: np.ndarray, M_model: np.ndarray,
                  FF_plant=None, FF_model=None):
        """One deterministic closed-loop trajectory (SIM-07)."""
        n_out, n_knob = self.n_out, self.n_knob
        A_d = self.plant.A_d if self.plant.A_d is not None else \
            self.plant.ar_coef * np.eye(n_out)
        c = self.controller

        d = rng.standard_normal(n_out) * self.plant.process_noise_std
        drift = np.zeros(n_out)
        d_hat = np.zeros(n_out)                       # MHE state estimate
        P = np.eye(n_out)                             # estimate covariance
        Q = c.mhe_Q * (self.plant.process_noise_std ** 2 + self.plant.drift_std ** 2) * np.eye(n_out)
        R = c.mhe_R * (self.plant.meas_noise_std ** 2) * np.eye(n_out)

        M_pinv = np.linalg.pinv(M_model)
        y = np.zeros((n_steps, n_out))
        u = np.zeros((n_steps, n_knob))
        state_est = np.zeros((n_steps, n_out))
        innovation = np.zeros((n_steps, n_out))
        disturbance = np.zeros((n_steps, n_out))
        n_ff = 0 if self.FF_gain is None else self.FF_gain.shape[1]
        ff_inputs = np.zeros((n_steps, n_ff)) if n_ff else np.zeros((n_steps, 0))

        u_prev = self.u0.copy()
        for t in range(n_steps):
            drift = drift + self.plant.drift_std * rng.standard_normal(n_out)
            d = A_d @ d + self.plant.process_noise_std * rng.standard_normal(n_out)
            dist = d + drift
            disturbance[t] = dist

            ff_contrib_plant = np.zeros(n_out)
            ff_contrib_model = np.zeros(n_out)
            if n_ff:
                ff = self.plant.ff_disturbance_std * rng.standard_normal(n_ff)
                ff_inputs[t] = ff
                ff_contrib_plant = FF_plant @ ff
                ff_contrib_model = FF_model @ ff

            # ---- MPC: choose move to drive predicted y toward target -----
            # predicted next disturbance (model) = A_d_model d_hat (use plant A_d)
            d_pred = A_d @ d_hat
            # desired control contribution: target - d_pred - ff_model
            desired = self.targets - d_pred - ff_contrib_model
            # finite-horizon LQ first-move (analytic, then clip): with static
            # gain over the horizon the optimum balances tracking vs move cost.
            #   u* = argmin Q||M(u-u0) - desired||^2 + R||u - u_prev||^2
            G = M_model
            QcI = c.mpc_Q
            RcI = c.mpc_R
            H = QcI * (G.T @ G) + RcI * np.eye(n_knob)
            rhs = QcI * (G.T @ desired) + RcI * (u_prev - self.u0)
            du = np.linalg.solve(H, rhs)               # = u - u0
            u_cmd = self.u0 + du
            # move/rate limit (SIM-04)
            step = np.clip(u_cmd - u_prev, -c.move_limit, c.move_limit)
            u_t = u_prev + step
            u[t] = u_t

            # ---- plant output -------------------------------------------
            meas_noise = self.plant.meas_noise_std * rng.standard_normal(n_out)
            y_t = dist + ff_contrib_plant + M_plant @ (u_t - self.u0) + meas_noise
            y[t] = y_t

            # ---- MHE: Kalman update of disturbance estimate (sampling) ----
            measured = (t % max(c.sampling_rate, 1)) == 0
            if measured:
                innov = y_t - (M_model @ (u_t - self.u0) + ff_contrib_model) - A_d @ d_hat
                innovation[t] = innov
                P_pred = A_d @ P @ A_d.T + Q
                S = P_pred + R
                K = P_pred @ np.linalg.inv(S)
                d_hat = A_d @ d_hat + K @ innov
                P = (np.eye(n_out) - K) @ P_pred
            else:
                d_hat = A_d @ d_hat
                P = A_d @ P @ A_d.T + Q
            state_est[t] = d_hat
            u_prev = u_t

        return y, u, state_est, innovation, disturbance, ff_inputs

    # ------------------------------------------------------------------ #
    def simulate(self, n_steps: int = 300, n_mc: Optional[int] = None) -> SimulationResult:
        """Run both arms; propagate model error via Monte-Carlo (SIM-01/02).

        ``n_mc`` Monte-Carlo replicates (drawing ``M``/``FF`` from their relative
        uncertainty) produce metric uncertainty bands; with ``n_mc=None`` and no
        uncertainty a single deterministic run is used.
        """
        delay = max(self.controller.metrology_delay, 1)
        if n_mc is None:
            n_mc = 30 if (self.rel_uncertainty_M > 0 or self.rel_uncertainty_FF > 0) else 1

        metrics_on_samples = []
        metrics_off_samples = []
        last = None
        for s in range(n_mc):
            rng = np.random.default_rng(self.seed + s)
            # plant gain is truth; the controller's *model* gain carries the error:
            # a systematic scale error (model_gain_scale) plus optional random error.
            M_plant = self.M
            M_model = self.M * self.model_gain_scale
            if self.rel_uncertainty_M > 0:
                M_model = M_model * (1.0 + self.rel_uncertainty_M *
                                     rng.standard_normal(self.M.shape))
            FF_plant = self.FF_gain
            FF_model = self.FF_gain
            if self.FF_gain is not None and self.rel_uncertainty_FF > 0:
                FF_model = self.FF_gain * (1.0 + self.rel_uncertainty_FF *
                                           rng.standard_normal(self.FF_gain.shape))
            y_on, u, est, innov, dist, ff = self._run_once(
                n_steps, rng, M_plant, M_model, FF_plant, FF_model)
            # ---- no-control arm via shared counterfactual service (SIM-01) ----
            cf = reconstruct(y_on, M_plant, u, self.u0, u0_source="simulator baseline")
            y_off = cf.y_nocontrol
            m_on = compute_metrics(y_on, self.targets, self.lsl, self.usl, delay)
            m_off = compute_metrics(y_off, self.targets, self.lsl, self.usl, delay)
            metrics_on_samples.append(m_on)
            metrics_off_samples.append(m_off)
            last = (y_on, y_off, u, est, innov, dist, ff)

        metrics_on = _aggregate_metrics(metrics_on_samples)
        metrics_off = _aggregate_metrics(metrics_off_samples)
        y_on, y_off, u, est, innov, dist, ff = last
        return SimulationResult(
            y_on=y_on, y_off=y_off, u=u, state_est=est, innovation=innov,
            disturbance=dist, ff_inputs=ff, metrics_on=metrics_on,
            metrics_off=metrics_off,
            config={"n_mc": n_mc, "rel_uncertainty_M": self.rel_uncertainty_M,
                    "controller": self.controller.__dict__,
                    "plant": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                              for k, v in self.plant.__dict__.items()}},
        )


def _aggregate_metrics(samples: List[Metrics]) -> Metrics:
    if len(samples) == 1:
        return samples[0]
    stack = lambda attr: np.stack([getattr(m, attr) for m in samples])
    out = Metrics(
        std=stack("std").mean(0),
        mean_off_target=stack("mean_off_target").mean(0),
        cpk=stack("cpk").mean(0),
        pct_oos=stack("pct_oos").mean(0),
        harris=stack("harris").mean(0),
    )
    out.bands = {
        "cpk_p05": np.nanpercentile(stack("cpk"), 5, axis=0),
        "cpk_p95": np.nanpercentile(stack("cpk"), 95, axis=0),
        "std_p05": np.nanpercentile(stack("std"), 5, axis=0),
        "std_p95": np.nanpercentile(stack("std"), 95, axis=0),
        "harris_p05": np.nanpercentile(stack("harris"), 5, axis=0),
        "harris_p95": np.nanpercentile(stack("harris"), 95, axis=0),
    }
    return out
