"""Component 4 — Controller Optimizer (§8, OPT-01..05).

Iteratively search controller/estimator configurations to find the
best-performing one, using a selectable optimization algorithm and the
simulator (§7) as the objective.  Because the simulator returns metric
distributions, the objective is *robust* (risk-adjusted), so the chosen
configuration is not tuned to one favorable noise realization.  The optimizer
reports objective sensitivity per dimension and rejects configurations the
diagnostics flag as unstable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize as _scipy_minimize

from ..foundations.gp import GaussianProcess
from .component3_simulator import Simulator, ControllerConfig


@dataclass
class SearchDimension:
    name: str          # e.g. 'mpc_R', 'mhe_Q', 'move_limit'
    low: float
    high: float
    log: bool = False  # search in log-space (good for Q/R ratios)

    def to_unit(self, value: float) -> float:
        if self.log:
            return (np.log(value) - np.log(self.low)) / (np.log(self.high) - np.log(self.low))
        return (value - self.low) / (self.high - self.low)

    def from_unit(self, u: float) -> float:
        u = float(np.clip(u, 0.0, 1.0))
        if self.log:
            return float(np.exp(np.log(self.low) + u * (np.log(self.high) - np.log(self.low))))
        return float(self.low + u * (self.high - self.low))


@dataclass
class OptimizationResult:
    best_config: Dict[str, float]
    best_objective: float
    trace: List[Tuple[Dict[str, float], float]]
    sensitivity: Dict[str, float]
    n_evals: int
    rejected_unstable: int
    algorithm: str

    def to_dict(self):
        return {
            "best_config": self.best_config,
            "best_objective": float(self.best_objective),
            "sensitivity": self.sensitivity,
            "n_evals": self.n_evals,
            "rejected_unstable": self.rejected_unstable,
            "algorithm": self.algorithm,
            "trace_objectives": [o for _, o in self.trace],
        }


class ControllerOptimizer:
    """Component 4.  Outer optimization loop wrapping the simulator (OPT-01)."""

    def __init__(
        self,
        simulator_factory: Callable[[ControllerConfig], Simulator],
        dimensions: List[SearchDimension],
        objective_metric: str = "std",
        risk_lambda: float = 1.0,
        n_steps: int = 250,
        n_mc: int = 8,
        stability_std_blowup: float = 5.0,
        seed: int = 0,
    ):
        """``simulator_factory`` builds a Simulator from a ControllerConfig.

        ``objective_metric`` in {'std','cpk','harris','mean_off_target'}.
        ``risk_lambda`` controls the robust objective: mean + λ·spread for
        minimized metrics (OPT-03).
        """
        self.simulator_factory = simulator_factory
        self.dims = dimensions
        self.objective_metric = objective_metric
        self.risk_lambda = risk_lambda
        self.n_steps = n_steps
        self.n_mc = n_mc
        self.stability_std_blowup = stability_std_blowup
        self.seed = seed
        self._trace: List[Tuple[Dict[str, float], float]] = []
        self._rejected = 0

    # ------------------------------------------------------------------ #
    def _config_from_vector(self, x_unit: np.ndarray) -> ControllerConfig:
        cfg = ControllerConfig()
        for d, u in zip(self.dims, x_unit):
            setattr(cfg, d.name, d.from_unit(u))
        return cfg

    def _robust_objective(self, x_unit: np.ndarray) -> float:
        cfg = self._config_from_vector(x_unit)
        sim = self.simulator_factory(cfg)
        res = sim.simulate(n_steps=self.n_steps, n_mc=self.n_mc)
        m_on = res.metrics_on

        # ---- stability rejection (OPT-05) ----------------------------
        off_std = np.nanmean(res.metrics_off.std)
        on_std = np.nanmean(m_on.std)
        if not np.isfinite(on_std) or on_std > self.stability_std_blowup * off_std:
            self._rejected += 1
            self._trace.append((self._readable(cfg), np.inf))
            return 1e6  # over-correction / instability => infeasible (OPT-05)

        # ---- robust metric (OPT-03) ----------------------------------
        if self.objective_metric == "std":
            base = np.nanmean(m_on.std)
            spread = self._spread(m_on, "std")
            val = base + self.risk_lambda * spread
        elif self.objective_metric == "cpk":
            base = np.nanmean(m_on.cpk)
            spread = self._spread(m_on, "cpk")
            val = -(base - self.risk_lambda * spread)          # maximize Cpk
        elif self.objective_metric == "harris":
            val = float(np.nanmean(np.abs(m_on.harris - 1.0)))  # drive Harris->1
        elif self.objective_metric == "mean_off_target":
            val = float(np.nanmean(m_on.mean_off_target))
        else:
            raise ValueError(f"unknown metric {self.objective_metric}")
        self._trace.append((self._readable(cfg), float(val)))
        return float(val)

    def _spread(self, metrics, attr) -> float:
        band_lo = metrics.bands.get(f"{attr}_p05")
        band_hi = metrics.bands.get(f"{attr}_p95")
        if band_lo is None or band_hi is None:
            return 0.0
        return float(np.nanmean(band_hi - band_lo))

    def _readable(self, cfg: ControllerConfig) -> Dict[str, float]:
        return {d.name: float(getattr(cfg, d.name)) for d in self.dims}

    # ------------------------------------------------------------------ #
    def optimize(self, algorithm: str = "bayesian", budget: int = 40) -> OptimizationResult:
        """Run the selected algorithm (OPT-02)."""
        self._trace = []
        self._rejected = 0
        rng = np.random.default_rng(self.seed)
        ndim = len(self.dims)

        if algorithm == "random":
            X = rng.random((budget, ndim))
            ys = [self._robust_objective(x) for x in X]
            best_i = int(np.argmin(ys))
            best_x = X[best_i]
        elif algorithm == "grid":
            per = max(int(round(budget ** (1.0 / ndim))), 2)
            axes = [np.linspace(0, 1, per) for _ in range(ndim)]
            grid = np.array(np.meshgrid(*axes)).reshape(ndim, -1).T
            ys = [self._robust_objective(x) for x in grid]
            best_i = int(np.argmin(ys))
            best_x = grid[best_i]
        elif algorithm == "nelder_mead":
            x0 = rng.random(ndim)
            res = _scipy_minimize(self._robust_objective, x0, method="Nelder-Mead",
                                  options={"maxfev": budget, "xatol": 1e-3, "fatol": 1e-4})
            best_x = np.clip(res.x, 0, 1)
        elif algorithm == "evolutionary":
            best_x = self._evolution(rng, ndim, budget)
        elif algorithm == "bayesian":
            best_x = self._bayesian(rng, ndim, budget)
        else:
            raise ValueError(f"unknown algorithm {algorithm!r}")

        best_obj = self._robust_objective(best_x)
        sensitivity = self._sensitivity(best_x)
        return OptimizationResult(
            best_config=self._readable(self._config_from_vector(best_x)),
            best_objective=best_obj,
            trace=list(self._trace),
            sensitivity=sensitivity,
            n_evals=len(self._trace),
            rejected_unstable=self._rejected,
            algorithm=algorithm,
        )

    # -- simple (mu+lambda) evolution strategy --------------------------
    def _evolution(self, rng, ndim, budget):
        pop = rng.random((6, ndim))
        fit = np.array([self._robust_objective(x) for x in pop])
        evals = len(pop)
        sigma = 0.25
        while evals < budget:
            parent = pop[int(np.argmin(fit))]
            children = np.clip(parent + sigma * rng.standard_normal((4, ndim)), 0, 1)
            cfit = np.array([self._robust_objective(c) for c in children])
            evals += len(children)
            allx = np.vstack([pop, children])
            allf = np.concatenate([fit, cfit])
            keep = np.argsort(allf)[:6]
            pop, fit = allx[keep], allf[keep]
            sigma *= 0.9
        return pop[int(np.argmin(fit))]

    # -- Bayesian optimization with GP surrogate + EI -------------------
    def _bayesian(self, rng, ndim, budget):
        n_init = min(max(2 * ndim, 4), budget)
        X = rng.random((n_init, ndim))
        y = np.array([self._robust_objective(x) for x in X])
        for _ in range(budget - n_init):
            gp = GaussianProcess(length_scale=0.3, signal_var=float(np.var(y) + 1e-6),
                                 noise_var=1e-4)
            gp.fit(X, y)
            cand = rng.random((400, ndim))
            ei = gp.expected_improvement(cand, best=float(np.min(y)), maximize=False)
            x_next = cand[int(np.argmax(ei))]
            y_next = self._robust_objective(x_next)
            X = np.vstack([X, x_next])
            y = np.append(y, y_next)
        return X[int(np.argmin(y))]

    # -- sensitivity report (OPT-04) ------------------------------------
    def _sensitivity(self, x_center: np.ndarray, delta: float = 0.15) -> Dict[str, float]:
        """One-at-a-time objective sensitivity around the optimum (OPT-04).

        Distinguishes dimensions the performance is genuinely sensitive to from
        those it is flat in — itself a noise-floor statement about which tuning
        choices matter.
        """
        base = self._robust_objective(x_center)
        sens = {}
        for j, d in enumerate(self.dims):
            xp = x_center.copy(); xp[j] = np.clip(xp[j] + delta, 0, 1)
            xm = x_center.copy(); xm[j] = np.clip(xm[j] - delta, 0, 1)
            fp = self._robust_objective(xp)
            fm = self._robust_objective(xm)
            sens[d.name] = float((abs(fp - base) + abs(fm - base)) / 2.0)
        return sens
