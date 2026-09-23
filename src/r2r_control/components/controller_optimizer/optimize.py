"""Component 4 — Controller Optimizer (§8, OPT-01..05).

Wraps the simulator in an outer optimization loop. Components never import each
other (§11/§16), so the optimizer takes a *simulator factory*: a callable that
maps a config dict ``{name: value}`` to an object with
``.simulate(n_steps, n_mc)`` returning ``.metrics_on`` / ``.metrics_off``. The
pipeline (or the caller) supplies a factory that builds the real Simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
from scipy.optimize import minimize as _scipy_minimize

from ...core.gp import GaussianProcess


@dataclass
class SearchDimension:
    name: str
    low: float
    high: float
    log: bool = False

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
        return {"best_config": self.best_config, "best_objective": float(self.best_objective),
                "sensitivity": self.sensitivity, "n_evals": self.n_evals,
                "rejected_unstable": self.rejected_unstable, "algorithm": self.algorithm,
                "trace_objectives": [o for _, o in self.trace]}


class ControllerOptimizer:
    """Component 4. Outer optimization loop wrapping the simulator (OPT-01)."""

    def __init__(self, simulator_factory: Callable[[Dict[str, float]], object],
                 dimensions: List[SearchDimension], objective_metric: str = "std",
                 risk_lambda: float = 1.0, n_steps: int = 250, n_mc: int = 8,
                 stability_std_blowup: float = 5.0, seed: int = 0):
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

    def _readable(self, x_unit: np.ndarray) -> Dict[str, float]:
        return {d.name: d.from_unit(u) for d, u in zip(self.dims, x_unit)}

    def _robust_objective(self, x_unit: np.ndarray) -> float:
        """Objective under model uncertainty (OPT-02/04); unstable configs rejected (OPT-05)."""
        cfg = self._readable(x_unit)
        res = self.simulator_factory(cfg).simulate(n_steps=self.n_steps, n_mc=self.n_mc)
        m_on = res.metrics_on
        on_std = np.nanmean(m_on.std)
        if not np.isfinite(on_std) or on_std > self.stability_std_blowup * np.nanmean(res.metrics_off.std):
            self._rejected += 1
            self._trace.append((cfg, np.inf))
            return 1e6
        if self.objective_metric == "std":
            val = on_std + self.risk_lambda * self._spread(m_on, "std")
        elif self.objective_metric == "cpk":
            val = -(np.nanmean(m_on.cpk) - self.risk_lambda * self._spread(m_on, "cpk"))
        elif self.objective_metric == "harris":
            val = float(np.nanmean(np.abs(m_on.harris - 1.0)))
        elif self.objective_metric == "mean_off_target":
            val = float(np.nanmean(m_on.mean_off_target))
        else:
            raise ValueError(f"unknown metric {self.objective_metric!r}")
        self._trace.append((cfg, float(val)))
        return float(val)

    @staticmethod
    def _spread(metrics, attr) -> float:
        lo, hi = metrics.bands.get(f"{attr}_p05"), metrics.bands.get(f"{attr}_p95")
        return 0.0 if lo is None or hi is None else float(np.nanmean(hi - lo))

    def optimize(self, algorithm: str = "bayesian", budget: int = 40) -> OptimizationResult:
        self._trace, self._rejected = [], 0
        rng = np.random.default_rng(self.seed)
        ndim = len(self.dims)
        if algorithm == "random":
            X = rng.random((budget, ndim))
            best_x = X[int(np.argmin([self._robust_objective(x) for x in X]))]
        elif algorithm == "grid":
            per = max(int(round(budget ** (1.0 / ndim))), 2)
            grid = np.array(np.meshgrid(*[np.linspace(0, 1, per)] * ndim)).reshape(ndim, -1).T
            best_x = grid[int(np.argmin([self._robust_objective(x) for x in grid]))]
        elif algorithm == "nelder_mead":
            res = _scipy_minimize(self._robust_objective, rng.random(ndim), method="Nelder-Mead",
                                  options={"maxfev": budget, "xatol": 1e-3, "fatol": 1e-4})
            best_x = np.clip(res.x, 0, 1)
        elif algorithm == "evolutionary":
            best_x = self._evolution(rng, ndim, budget)
        elif algorithm == "bayesian":
            best_x = self._bayesian(rng, ndim, budget)
        else:
            raise ValueError(f"unknown algorithm {algorithm!r}")
        best_obj = self._robust_objective(best_x)
        return OptimizationResult(best_config=self._readable(best_x), best_objective=best_obj,
                                  trace=list(self._trace), sensitivity=self._sensitivity(best_x),
                                  n_evals=len(self._trace), rejected_unstable=self._rejected,
                                  algorithm=algorithm)

    def _evolution(self, rng, ndim, budget):
        pop = rng.random((6, ndim))
        fit = np.array([self._robust_objective(x) for x in pop])
        evals, sigma = len(pop), 0.25
        while evals < budget:
            children = np.clip(pop[int(np.argmin(fit))] + sigma * rng.standard_normal((4, ndim)), 0, 1)
            cfit = np.array([self._robust_objective(c) for c in children])
            evals += len(children)
            allx, allf = np.vstack([pop, children]), np.concatenate([fit, cfit])
            keep = np.argsort(allf)[:6]
            pop, fit = allx[keep], allf[keep]
            sigma *= 0.9
        return pop[int(np.argmin(fit))]

    def _bayesian(self, rng, ndim, budget):
        n_init = min(max(2 * ndim, 4), budget)
        X = rng.random((n_init, ndim))
        y = np.array([self._robust_objective(x) for x in X])
        for _ in range(budget - n_init):
            gp = GaussianProcess(length_scale=0.3, signal_var=float(np.var(y) + 1e-6),
                                 noise_var=1e-4)
            gp.fit(X, y)
            cand = rng.random((400, ndim))
            x_next = cand[int(np.argmax(gp.expected_improvement(cand, best=float(np.min(y)),
                                                                maximize=False)))]
            X = np.vstack([X, x_next])
            y = np.append(y, self._robust_objective(x_next))
        return X[int(np.argmin(y))]

    def _sensitivity(self, x_center: np.ndarray, delta: float = 0.15) -> Dict[str, float]:
        """Objective sensitivity to each tuning dimension (OPT-03)."""
        base = self._robust_objective(x_center)
        sens = {}
        for j, d in enumerate(self.dims):
            xp, xm = x_center.copy(), x_center.copy()
            xp[j] = np.clip(xp[j] + delta, 0, 1)
            xm[j] = np.clip(xm[j] - delta, 0, 1)
            sens[d.name] = float((abs(self._robust_objective(xp) - base) +
                                  abs(self._robust_objective(xm) - base)) / 2.0)
        return sens
