"""Component 2 — Experiment Planner (§6, EP-01..08)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .gap import GapReport, diagnose_gap
from .seed_design import seed_design
from .infill import sequential_infill
from .value import ExperimentProposal
from .loop_status import closed_loop_excitation_discount, residual_excitation
from .approval import APPROVAL_REQUIRED


@dataclass
class PlannerConfig:
    """Open design decisions declared as configuration, not hidden defaults (EP-08)."""

    model_form: str = "linear+interactions"
    primary_objective: str = "D"
    px_weighting: str = "uniform"
    loop_status: str = "open"             # loop status during experiments (A6/E9)
    rejection_fraction: float = 0.6       # share of excitation the loop cancels if closed
    budget: int = 8
    level_low: float = -1.0
    level_high: float = 1.0
    candidate_pool: int = 400
    stopping_info_gain: float = 1e-3
    integer_dims: Optional[List[int]] = None   # E11


@dataclass
class PlannerResult:
    gap: GapReport
    proposals: List[ExperimentProposal]
    gate_log: List[str]
    config: Dict
    stopped_reason: str

    def to_dict(self):
        return {"gap": self.gap.to_dict(), "proposals": [p.to_dict() for p in self.proposals],
                "gate_log": self.gate_log, "config": self.config,
                "stopped_reason": self.stopped_reason}


class ExperimentPlanner:
    """Component 2. Gate-based workflow producing ranked, justified proposals (EP-07)."""

    def __init__(self, config: Optional[PlannerConfig] = None, seed: int = 0):
        self.cfg = config or PlannerConfig()
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    @property
    def excitation_discount(self) -> float:
        return closed_loop_excitation_discount(self.cfg.loop_status, self.cfg.rejection_fraction)

    def diagnose_gap(self, X_passive, feature_names, adjustable) -> GapReport:
        return diagnose_gap(X_passive, feature_names, adjustable)

    def seed_design(self, adjustable, n_runs=None, objective=None) -> np.ndarray:
        return seed_design(len(adjustable), objective or self.cfg.primary_objective,
                           self.cfg.model_form, n_runs or max(self.cfg.budget, len(adjustable) + 1),
                           self.cfg.level_low, self.cfg.level_high, self.cfg.candidate_pool,
                           self.rng, integer_dims=self.cfg.integer_dims)

    def sequential_infill(self, X_existing, adjustable, n_new, acquisition="alc",
                          y_existing=None) -> List[ExperimentProposal]:
        return sequential_infill(X_existing, adjustable, n_new, self.cfg.level_low,
                                 self.cfg.level_high, self.cfg.candidate_pool, self.rng,
                                 acquisition=acquisition, y_existing=y_existing,
                                 integer_dims=self.cfg.integer_dims,
                                 discount=self.excitation_discount)

    def plan(self, X_passive, feature_names, adjustable, y_passive=None) -> PlannerResult:
        """framing -> diagnosis -> seed -> infill -> stopping (EP-07); manual approval (EP-06)."""
        log = [f"framing: objective={self.cfg.primary_objective}, loop_status="
               f"{self.cfg.loop_status} (excitation discount {self.excitation_discount:.2f}, "
               f"A6), model_form={self.cfg.model_form} (EP-08)"]
        gap = self.diagnose_gap(X_passive, feature_names, adjustable)
        log.append(f"passive-data diagnosis: {gap.confounding_summary}")
        seed = self.seed_design(adjustable)
        log.append(f"seed design: {len(seed)} runs, {self.cfg.primary_objective}-optimal")
        infill = self.sequential_infill(seed, adjustable, n_new=self.cfg.budget)
        stopped, kept = "budget exhausted", []
        for p in infill:
            if p.information_gain < self.cfg.stopping_info_gain:
                stopped = (f"marginal information gain {p.information_gain:.2e} below "
                           f"threshold {self.cfg.stopping_info_gain:.1e}")
                break
            kept.append(p)
        proposals = kept or infill[:1]
        log.append(f"sequential infill: {len(proposals)} experiments proposed; stopped "
                   f"because {stopped}")
        log.append(APPROVAL_REQUIRED)
        return PlannerResult(gap=gap, proposals=proposals, gate_log=log,
                             config=dict(self.cfg.__dict__), stopped_reason=stopped)


__all__ = ["ExperimentPlanner", "PlannerConfig", "PlannerResult", "ExperimentProposal",
           "GapReport", "closed_loop_excitation_discount", "residual_excitation"]
