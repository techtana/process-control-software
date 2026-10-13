"""Gate-based pipeline orchestration (EP-07, §IF-01..06).

The pipeline is the only place that wires components together; the components
never import each other (§11). It follows the artifact-mediated interfaces:
identification -> gap diagnosis / planning -> simulation -> (optimization) ->
diagnosis -> regression.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..config.schema import RunConfig
from ..contracts.schema import EventTable
from ..components.fb_identification import FBIdentifier, FBModel
from ..components.experiment_planner import ExperimentPlanner, PlannerConfig
from ..components.simulator import Simulator, PlantConfig, ControllerConfig
from ..components.controller_optimizer import ControllerOptimizer, SearchDimension
from ..components.diagnostics import DiagnosticsEngine
from ..components.regression_machine import RegressionMachine


@dataclass
class PipelineResult:
    fb: FBModel
    plan: object = None
    simulation: object = None
    optimization: object = None
    diagnosis: object = None
    regression: object = None
    gate_log: List[str] = field(default_factory=list)

    def to_dict(self):
        def d(x):
            return None if x is None else x.to_dict()
        return {"fb": self.fb.to_dict(), "plan": d(self.plan),
                "optimization": d(self.optimization), "diagnosis": d(self.diagnosis),
                "regression": d(self.regression), "gate_log": self.gate_log}


class GatePipeline:
    """End-to-end orchestration honoring the artifact-mediated interfaces (§11)."""

    def __init__(self, config: Optional[RunConfig] = None):
        self.cfg = config or RunConfig()

    def simulator_factory(self, fb: FBModel):
        """Build the factory the optimizer uses (dependency injection, §11)."""
        if self.cfg.targets is None:
            raise ValueError("RunConfig.targets/lsl/usl are required to simulate (§13.3)")
        t = np.asarray(self.cfg.targets, dtype=float)
        lsl = np.asarray(self.cfg.lsl, dtype=float)
        usl = np.asarray(self.cfg.usl, dtype=float)

        def factory(cdict: Dict[str, float]) -> Simulator:
            ctrl = ControllerConfig()
            for k, v in cdict.items():
                setattr(ctrl, k, v)
            return Simulator(M=fb.M, targets=t, lsl=lsl, usl=usl, u0=fb.u0_static(),
                             plant=PlantConfig(), controller=ctrl,
                             rel_uncertainty_M=self.cfg.rel_uncertainty_M, seed=self.cfg.seed)
        return factory

    def run(self, inline: EventTable, doe: Optional[EventTable] = None,
            doe_age_days: float = 0.0, run_planner: bool = True, run_simulation: bool = True,
            run_optimizer: bool = False) -> PipelineResult:
        log = [f"framing: seed={self.cfg.seed}, planner_objective={self.cfg.planner_objective}, "
               f"loop_status={self.cfg.loop_status} (§13 explicit config)"]

        # Gate 1 — FB identification (IF-01)
        fb = FBIdentifier(estimator=self.cfg.estimator_fb, seed=self.cfg.seed).identify(
            inline, doe, doe_age_days=doe_age_days,
            allow_time_varying_u0=self.cfg.allow_time_varying_u0)
        log.append(f"gate 1 — identification: gain rel-uncertainty {fb.relative_uncertainty():.3f}, "
                   f"effective rank {fb.identifiable.effective_rank}/"
                   f"{fb.identifiable.n_directions}")
        result = PipelineResult(fb=fb, gate_log=log)

        # Gate 2 — gap diagnosis -> experiment plan (IF-02)
        if run_planner:
            planner = ExperimentPlanner(PlannerConfig(
                primary_objective=self.cfg.planner_objective, loop_status=self.cfg.loop_status,
                budget=self.cfg.budget, integer_dims=self.cfg.integer_dims), seed=self.cfg.seed)
            result.plan = planner.plan(inline.regressors(), inline.regressor_names,
                                       inline.knob_cols_used)
            log.append(f"gate 2 — planner: {result.plan.gap.confounding_summary}; "
                       f"{len(result.plan.proposals)} experiments proposed")

        # Gates 3-5 — simulate -> (optimize) -> diagnose (IF-03..05)
        if run_simulation and self.cfg.targets is not None:
            factory = self.simulator_factory(fb)
            best = {"mpc_R": 0.05, "move_limit": 1.0}
            if run_optimizer:
                opt = ControllerOptimizer(factory, [SearchDimension("mpc_R", 1e-3, 1.0, log=True),
                                                    SearchDimension("move_limit", 0.2, 1.5)],
                                          objective_metric="std", n_steps=200, n_mc=4,
                                          seed=self.cfg.seed)
                result.optimization = opt.optimize(algorithm="bayesian", budget=18)
                best = result.optimization.best_config
                log.append(f"gate 4 — optimizer: best robust std "
                           f"{result.optimization.best_objective:.3f} at {best}")
            sr = factory(best).simulate(n_steps=300)
            result.simulation = sr
            result.diagnosis = DiagnosticsEngine().diagnose(
                sr.y_on, sr.u, fb.M, state_est=sr.state_est, ff_inputs=sr.ff_inputs,
                innovation=sr.innovation, monitor_gain_drift=True)
            log.append(f"gate 3/5 — simulation + diagnosis: "
                       f"{result.diagnosis.achievability['verdict']}")

        # Gate 6 — all-data regression (IF-01)
        result.regression = RegressionMachine(seed=self.cfg.seed).fit(
            inline, inline.y_cols[0], M=fb.M, u0=fb.u0, target_kind=self.cfg.regression_target,
            adjustable_features=inline.knob_cols_used,
            rel_uncertainty_M=self.cfg.rel_uncertainty_M)
        log.append(f"gate 6 — regression: estimator {result.regression.estimator}, "
                   f"FF candidates {result.regression.feedforward_candidates}, quarantined "
                   f"{result.regression.quarantined_sensors}")
        return result
