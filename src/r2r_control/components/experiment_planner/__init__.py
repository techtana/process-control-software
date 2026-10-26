"""Component 2 — Experiment Planner (§6, EP-01..08)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .gap import GapReport, diagnose_gap
from .seed_design import seed_design
from .infill import sequential_infill, gap_alignment
from .space import CodedSpace
from .value import ExperimentProposal
from .loop_status import closed_loop_excitation_discount, residual_excitation
from .approval import APPROVAL_REQUIRED


@dataclass
class PlannerConfig:
    """Open design decisions declared as configuration, not hidden defaults (EP-08)."""

    model_form: str = "linear+interactions"
    primary_objective: str = "D"
    px_weighting: str = "uniform"         # p(x) for I-optimality / ALC: 'uniform' | 'central'
    loop_status: str = "open"             # loop status during experiments (A6/E9)
    rejection_fraction: float = 0.6       # share of excitation the loop cancels if closed
    budget: int = 8
    level_low: float = -1.0
    level_high: float = 1.0
    candidate_pool: int = 400
    stopping_info_gain: float = 1e-3
    integer_dims: Optional[List[int]] = None   # E11
    categorical_levels: Optional[Dict[int, List[float]]] = None   # E11, coded levels
    acquisition: str = "alc"              # 'alc' | 'imse' | 'eig'
    gap_rtol: float = 1e-2                # singular-value ratio below which a direction is a gap
    gap_weight: float = 4.0               # acquisition boost for moves along gap directions
    knob_bounds: Optional[Dict[str, Tuple[float, float]]] = None   # engineering range per knob
    run_cost: float = 1.0                 # fixed cost per experiment
    knob_cost: Optional[Dict[str, float]] = None   # cost per unit coded excursion


@dataclass
class PlannerResult:
    gap: GapReport
    proposals: List[ExperimentProposal]
    gate_log: List[str]
    config: Dict
    stopped_reason: str
    seed_design: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    seed_design_engineering: Optional[np.ndarray] = None

    def to_dict(self):
        return {"gap": self.gap.to_dict(), "proposals": [p.to_dict() for p in self.proposals],
                "gate_log": self.gate_log, "config": self.config,
                "stopped_reason": self.stopped_reason,
                "seed_design": np.asarray(self.seed_design).tolist(),
                "seed_design_engineering": (None if self.seed_design_engineering is None
                                            else np.asarray(self.seed_design_engineering).tolist())}


class ExperimentPlanner:
    """Component 2. Gate-based workflow producing ranked, justified proposals (EP-07)."""

    def __init__(self, config: Optional[PlannerConfig] = None, seed: int = 0):
        self.cfg = config or PlannerConfig()
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    @property
    def excitation_discount(self) -> float:
        return closed_loop_excitation_discount(self.cfg.loop_status, self.cfg.rejection_fraction)

    def calibrate_rejection(self, commanded: np.ndarray, applied_residual: np.ndarray) -> float:
        """Set ``rejection_fraction`` from a measured surviving excitation share (A6/E9).

        ``commanded`` is the designed perturbation and ``applied_residual`` what
        actually reached the process after the controller reacted (e.g. from a
        pilot experiment). Replaces the configured guess with the measured value.
        """
        surviving = residual_excitation(commanded, applied_residual)
        self.cfg.rejection_fraction = float(1.0 - surviving)
        return self.cfg.rejection_fraction

    def space(self, adjustable: List[str]) -> CodedSpace:
        return CodedSpace(list(adjustable), self.cfg.level_low, self.cfg.level_high,
                          knob_bounds=self.cfg.knob_bounds, integer_dims=self.cfg.integer_dims,
                          categorical_levels=self.cfg.categorical_levels,
                          px_weighting=self.cfg.px_weighting)

    def diagnose_gap(self, X_passive, feature_names, adjustable,
                     target_features=None) -> GapReport:
        return diagnose_gap(X_passive, feature_names, adjustable, rtol=self.cfg.gap_rtol,
                            target_features=target_features)

    def seed_design(self, adjustable, n_runs=None, objective=None) -> np.ndarray:
        return seed_design(len(adjustable), objective or self.cfg.primary_objective,
                           self.cfg.model_form, n_runs or max(self.cfg.budget, len(adjustable) + 1),
                           self.cfg.level_low, self.cfg.level_high, self.cfg.candidate_pool,
                           self.rng, integer_dims=self.cfg.integer_dims,
                           space=self.space(adjustable))

    def sequential_infill(self, X_existing, adjustable, n_new, acquisition=None,
                          y_existing=None, gap: Optional[GapReport] = None,
                          purpose: str = "") -> List[ExperimentProposal]:
        V = gap.direction_matrix(list(adjustable)) if gap is not None else None
        return sequential_infill(X_existing, adjustable, n_new, self.cfg.level_low,
                                 self.cfg.level_high, self.cfg.candidate_pool, self.rng,
                                 acquisition=acquisition or self.cfg.acquisition,
                                 y_existing=y_existing, integer_dims=self.cfg.integer_dims,
                                 discount=self.excitation_discount,
                                 space=self.space(adjustable), gap_directions=V,
                                 gap_weight=self.cfg.gap_weight, base_cost=self.cfg.run_cost,
                                 knob_cost=self.cfg.knob_cost, purpose=purpose)

    def plan(self, X_passive, feature_names, adjustable, y_passive=None,
             target_features: Optional[List[str]] = None,
             X_existing: Optional[np.ndarray] = None,
             y_existing: Optional[np.ndarray] = None) -> PlannerResult:
        """framing -> diagnosis -> seed -> infill -> stopping (EP-07); manual approval (EP-06).

        ``target_features`` are knobs another component flagged as poorly
        identified (e.g. FB-04 low-confidence knobs); they join the gap.
        ``X_existing``/``y_existing`` are experiments already run (coded units)
        with their responses: they replace the seed design and the GP is fitted
        to the responses. ``y_passive`` is accepted for API compatibility and
        not used (passive responses are confounded by the loop).
        """
        adjustable = list(adjustable)
        space = self.space(adjustable)
        log = [f"framing: objective={self.cfg.primary_objective}, loop_status="
               f"{self.cfg.loop_status} (excitation discount {self.excitation_discount:.2f}, "
               f"A6), model_form={self.cfg.model_form}, p(x)={self.cfg.px_weighting} (EP-08)"]
        gap = self.diagnose_gap(X_passive, feature_names, adjustable, target_features)
        log.append(f"passive-data diagnosis: {gap.confounding_summary}")
        if gap.directions:
            purpose = (f"close identifiability gap ({len(gap.directions)} direction(s) over "
                       f"{gap.unidentifiable_features})")
            log.append(f"targeting: infill acquisition boosted x(1+{self.cfg.gap_weight:g}·"
                       f"alignment) toward the gap directions (KI-6)")
        else:
            purpose = "refine model (no identifiability gap in adjustable directions)"
            log.append("targeting: no identifiability gap in adjustable directions; proposals "
                       "are optional model-refinement runs, not gap-closing ones")

        if X_existing is not None and len(np.atleast_2d(X_existing)):
            base = np.atleast_2d(np.asarray(X_existing, dtype=float))
            seed = np.zeros((0, len(adjustable)))
            log.append(f"existing experiments: {len(base)} runs replace the seed design")
        else:
            seed = self.seed_design(adjustable)
            base = seed
            log.append(f"seed design: {len(seed)} runs, {self.cfg.primary_objective}-optimal")
        has_y = y_existing is not None and len(np.ravel(y_existing)) >= 3
        log.append("surrogate: GP hyperparameters fitted to observed responses (marginal "
                   "likelihood)" if has_y else
                   "surrogate: no responses yet, so the infill criterion is geometric "
                   "(space-filling, gap-weighted)")
        infill = self.sequential_infill(base, adjustable, n_new=self.cfg.budget,
                                        y_existing=y_existing if has_y else None, gap=gap,
                                        purpose=purpose)
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
        seed_eng = space.to_engineering(seed) if (self.cfg.knob_bounds and len(seed)) else None
        return PlannerResult(gap=gap, proposals=proposals, gate_log=log,
                             config=dict(self.cfg.__dict__), stopped_reason=stopped,
                             seed_design=seed, seed_design_engineering=seed_eng)


__all__ = ["ExperimentPlanner", "PlannerConfig", "PlannerResult", "ExperimentProposal",
           "GapReport", "CodedSpace", "closed_loop_excitation_discount", "residual_excitation",
           "gap_alignment"]
