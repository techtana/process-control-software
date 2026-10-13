"""Component 3 — MHE / MPC Controller Simulator (§7, SIM-01..07).

Simulates process outcomes with and without control, propagating FB/FF model
error by Monte Carlo. The no-control arm is the algebraic counterfactual from
the shared service (§CF). Built on the controller/ layer (MHE + MPC, with
out-of-sequence routing, A4/E7) and the metrics/ layer. Deterministic given a
seed (SIM-07).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ...controller.mhe.estimator import KalmanMHE
from ...controller.mhe.oosm import SlowBiasEstimator, route_measurement
from ...controller.mpc.controller import mpc_move
from ...core.counterfactual.reconstruct import reconstruct
from ...metrics.capability import Metrics, compute_metrics


@dataclass
class PlantConfig:
    A_d: Optional[np.ndarray] = None
    drift_std: float = 0.01
    ar_coef: float = 0.85
    process_noise_std: float = 0.3
    meas_noise_std: float = 0.2
    ff_disturbance_std: float = 0.5


@dataclass
class ControllerConfig:
    mhe_Q: float = 1.0
    mhe_R: float = 1.0
    mhe_horizon: int = 1
    mpc_Q: float = 1.0
    mpc_R: float = 0.1
    mpc_horizon: int = 5
    move_limit: float = 0.5
    sampling_rate: int = 1
    metrology_delay: int = 0


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
        return {"metrics_on": self.metrics_on.to_dict(),
                "metrics_off": self.metrics_off.to_dict(), "config": self.config}


class Simulator:
    """Component 3. Configurable MHE/MPC closed-loop simulator."""

    def __init__(self, M, targets, lsl, usl, u0=None, FF_gain=None,
                 plant: Optional[PlantConfig] = None,
                 controller: Optional[ControllerConfig] = None,
                 rel_uncertainty_M: float = 0.0, rel_uncertainty_FF: float = 0.0,
                 model_gain_scale: float = 1.0, seed: int = 0):
        self.M = np.asarray(M, dtype=float)
        self.n_out, self.n_knob = self.M.shape
        self.targets = np.asarray(targets, dtype=float)
        self.lsl = np.asarray(lsl, dtype=float)
        self.usl = np.asarray(usl, dtype=float)
        self.u0 = np.zeros(self.n_knob) if u0 is None else np.asarray(u0, dtype=float)
        self.FF_gain = None if FF_gain is None else np.asarray(FF_gain, dtype=float)
        self.plant = plant or PlantConfig()
        self.controller = controller or ControllerConfig()
        self.rel_uncertainty_M = rel_uncertainty_M
        self.rel_uncertainty_FF = rel_uncertainty_FF
        self.model_gain_scale = model_gain_scale
        self.seed = seed

    def _run_once(self, n_steps, rng, M_plant, M_model, FF_plant=None, FF_model=None):
        n_out, n_knob = self.n_out, self.n_knob
        A_d = self.plant.A_d if self.plant.A_d is not None else self.plant.ar_coef * np.eye(n_out)
        c = self.controller
        Q = c.mhe_Q * (self.plant.process_noise_std ** 2 + self.plant.drift_std ** 2) * np.eye(n_out)
        R = c.mhe_R * (self.plant.meas_noise_std ** 2) * np.eye(n_out)
        mhe = KalmanMHE(A_d, Q, R, d0=np.zeros(n_out), P0=np.eye(n_out))
        slow_bias = SlowBiasEstimator(n_out)
        # measurements delayed past the MHE horizon bypass the state update (A4/E7)
        route = route_measurement(c.metrology_delay, c.mhe_horizon)

        d = rng.standard_normal(n_out) * self.plant.process_noise_std
        drift = np.zeros(n_out)
        n_ff = 0 if self.FF_gain is None else self.FF_gain.shape[1]
        y = np.zeros((n_steps, n_out))
        u = np.zeros((n_steps, n_knob))
        state_est = np.zeros((n_steps, n_out))
        innovation = np.zeros((n_steps, n_out))
        disturbance = np.zeros((n_steps, n_out))
        ff_inputs = np.zeros((n_steps, n_ff))

        u_prev = self.u0.copy()
        for t in range(n_steps):
            drift = drift + self.plant.drift_std * rng.standard_normal(n_out)
            d = A_d @ d + self.plant.process_noise_std * rng.standard_normal(n_out)
            disturbance[t] = d + drift

            ff_plant = np.zeros(n_out)
            ff_model = np.zeros(n_out)
            if n_ff:
                ff = self.plant.ff_disturbance_std * rng.standard_normal(n_ff)
                ff_inputs[t] = ff
                ff_plant = FF_plant @ ff
                ff_model = FF_model @ ff

            desired = self.targets - (mhe.predict_state() + slow_bias.correction()) - ff_model
            u_t = mpc_move(M_model, desired, u_prev, self.u0, c.mpc_Q, c.mpc_R, c.move_limit)
            u[t] = u_t

            y_t = (disturbance[t] + ff_plant + M_plant @ (u_t - self.u0)
                   + self.plant.meas_noise_std * rng.standard_normal(n_out))
            y[t] = y_t

            if t % max(c.sampling_rate, 1) == 0:
                innov = mhe.innovation(y_t, M_model @ (u_t - self.u0), ff_model)
                innovation[t] = innov
                if route == "state_update":
                    mhe.update(innov)
                else:
                    mhe.propagate()
                    slow_bias.incorporate_late(innov)
            else:
                mhe.propagate()
            state_est[t] = mhe.d_hat + slow_bias.correction()
            u_prev = u_t
        return y, u, state_est, innovation, disturbance, ff_inputs

    def simulate(self, n_steps: int = 300, n_mc: Optional[int] = None) -> SimulationResult:
        delay = max(self.controller.metrology_delay, 1)
        if n_mc is None:
            n_mc = 30 if (self.rel_uncertainty_M > 0 or self.rel_uncertainty_FF > 0) else 1
        on_samples, off_samples, last = [], [], None
        for s in range(n_mc):
            rng = np.random.default_rng(self.seed + s)
            M_model = self.M * self.model_gain_scale
            if self.rel_uncertainty_M > 0:
                M_model = M_model * (1.0 + self.rel_uncertainty_M * rng.standard_normal(self.M.shape))
            FF_model = self.FF_gain
            if self.FF_gain is not None and self.rel_uncertainty_FF > 0:
                FF_model = self.FF_gain * (1.0 + self.rel_uncertainty_FF *
                                           rng.standard_normal(self.FF_gain.shape))
            y_on, u, est, innov, dist, ff = self._run_once(n_steps, rng, self.M, M_model,
                                                           self.FF_gain, FF_model)
            y_off = reconstruct(y_on, self.M, u, self.u0,
                                u0_source="simulator baseline").y_nocontrol
            on_samples.append(compute_metrics(y_on, self.targets, self.lsl, self.usl, delay))
            off_samples.append(compute_metrics(y_off, self.targets, self.lsl, self.usl, delay))
            last = (y_on, y_off, u, est, innov, dist, ff)
        y_on, y_off, u, est, innov, dist, ff = last
        return SimulationResult(
            y_on=y_on, y_off=y_off, u=u, state_est=est, innovation=innov, disturbance=dist,
            ff_inputs=ff, metrics_on=_aggregate(on_samples), metrics_off=_aggregate(off_samples),
            config={"n_mc": n_mc, "rel_uncertainty_M": self.rel_uncertainty_M,
                    "controller": dict(self.controller.__dict__),
                    "plant": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                              for k, v in self.plant.__dict__.items()}})


def _aggregate(samples: List[Metrics]) -> Metrics:
    if len(samples) == 1:
        return samples[0]

    def stack(attr):
        return np.stack([getattr(m, attr) for m in samples])

    out = Metrics(std=stack("std").mean(0), mean_off_target=stack("mean_off_target").mean(0),
                  cpk=stack("cpk").mean(0), pct_oos=stack("pct_oos").mean(0),
                  harris=stack("harris").mean(0))
    for attr in ("cpk", "std", "harris"):
        out.bands[f"{attr}_p05"] = np.nanpercentile(stack(attr), 5, axis=0)
        out.bands[f"{attr}_p95"] = np.nanpercentile(stack(attr), 95, axis=0)
    return out
