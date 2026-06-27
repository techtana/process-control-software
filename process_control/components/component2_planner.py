"""Component 2 — Experiment Planner (§6, EP-01..08).

Study the gap in current data, propose experiments that add the most
model-relevant information, prioritize them, and estimate each one's value — so
high-cost experiments are justified before a human approves them.  The planner
proposes on **adjustable inputs only**, supports D-/I-optimal seed designs and
GP-surrogate sequential infill, attaches a priority metric and estimated value
to every proposal, and operates under manual approval (it proposes, it does not
execute).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..foundations.gp import GaussianProcess
from ..foundations.excitation import vif, regressor_spectrum, unidentifiable_directions


@dataclass
class PlannerConfig:
    """Open design decisions declared as configuration, not hidden defaults (EP-08)."""

    model_form: str = "linear+interactions"
    primary_objective: str = "D"            # 'D' (param precision) | 'I' (prediction var)
    px_weighting: str = "uniform"           # input distribution p(x) for I-optimality
    loop_status: str = "open"               # loop status during experiments (open/closed)
    budget: int = 8                         # number of experiments
    budget_granularity: int = 1
    level_low: float = -1.0
    level_high: float = 1.0
    candidate_pool: int = 400
    stopping_info_gain: float = 1e-3        # marginal info-gain threshold (EP-07)


@dataclass
class ExperimentProposal:
    """A single proposed experiment with its justification (EP-05)."""

    setpoint: Dict[str, float]              # adjustable inputs only (EP-02)
    excites_directions: List[str]
    information_gain: float
    predicted_model_improvement: float
    priority: float
    feasible: bool = True
    cost: float = 1.0

    def to_dict(self):
        return {
            "setpoint": self.setpoint,
            "excites_directions": self.excites_directions,
            "information_gain": float(self.information_gain),
            "predicted_model_improvement": float(self.predicted_model_improvement),
            "priority": float(self.priority),
            "feasible": self.feasible,
            "cost": self.cost,
        }


@dataclass
class GapReport:
    """Quantified, localized gap from passive-data diagnosis (EP-01)."""

    unidentifiable_features: List[str]
    vif: Dict[str, float]
    effective_rank: int
    n_directions: int
    confounding_summary: str

    def to_dict(self):
        return {
            "unidentifiable_features": self.unidentifiable_features,
            "vif": self.vif,
            "effective_rank": self.effective_rank,
            "n_directions": self.n_directions,
            "confounding_summary": self.confounding_summary,
        }


@dataclass
class PlannerResult:
    gap: GapReport
    proposals: List[ExperimentProposal]
    gate_log: List[str]
    config: Dict
    stopped_reason: str

    def to_dict(self):
        return {
            "gap": self.gap.to_dict(),
            "proposals": [p.to_dict() for p in self.proposals],
            "gate_log": self.gate_log,
            "config": self.config,
            "stopped_reason": self.stopped_reason,
        }


class ExperimentPlanner:
    """Component 2.  Gate-based workflow producing ranked, justified proposals."""

    def __init__(self, config: Optional[PlannerConfig] = None, seed: int = 0):
        self.cfg = config or PlannerConfig()
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    # -- EP-01 : passive-data diagnosis ---------------------------------
    def diagnose_gap(self, X_passive: np.ndarray, feature_names: List[str],
                     adjustable: List[str]) -> GapReport:
        """Quantify and localize the gap from confounding measures (EP-01)."""
        v = vif(X_passive)
        spec = regressor_spectrum(X_passive, names=feature_names)
        unident = unidentifiable_directions(X_passive, names=feature_names)
        # only adjustable directions are actionable (EP-02)
        adj_set = set(adjustable)
        unident_feats = sorted({d for u in unident for d in u.dominant if d in adj_set})
        summary = (f"effective rank {spec.numerical_rank}/{len(feature_names)}, "
                   f"condition number {spec.condition_number:.1f}; "
                   f"{len(unident_feats)} adjustable directions unidentifiable")
        return GapReport(
            unidentifiable_features=unident_feats,
            vif={n: float(x) for n, x in zip(feature_names, v)},
            effective_rank=spec.numerical_rank,
            n_directions=len(feature_names),
            confounding_summary=summary,
        )

    # -- EP-03 : seed designs -------------------------------------------
    def seed_design(self, adjustable: List[str], n_runs: Optional[int] = None,
                    objective: Optional[str] = None) -> np.ndarray:
        """D- or I-optimal seed design via coordinate exchange (EP-03)."""
        objective = objective or self.cfg.primary_objective
        d = len(adjustable)
        n_runs = n_runs or max(self.cfg.budget, d + 1)
        pool = self.rng.uniform(self.cfg.level_low, self.cfg.level_high,
                                size=(self.cfg.candidate_pool, d))
        # model matrix builder
        def model_matrix(Z):
            return self._model_matrix(Z)
        # greedy start
        idx = list(self.rng.choice(len(pool), size=n_runs, replace=False))
        design = pool[idx]
        ref = self.rng.uniform(self.cfg.level_low, self.cfg.level_high, size=(200, d))
        Mref = model_matrix(ref)
        for _ in range(5):  # coordinate-exchange sweeps
            improved = False
            for i in range(n_runs):
                best_j, best_score = None, self._design_score(model_matrix(design),
                                                              Mref, objective)
                for j in range(len(pool)):
                    trial = design.copy()
                    trial[i] = pool[j]
                    sc = self._design_score(model_matrix(trial), Mref, objective)
                    if sc > best_score + 1e-9:
                        best_score, best_j = sc, j
                if best_j is not None:
                    design[i] = pool[best_j]
                    improved = True
            if not improved:
                break
        return design

    def _model_matrix(self, Z: np.ndarray) -> np.ndarray:
        Z = np.atleast_2d(Z)
        cols = [np.ones(len(Z)), *[Z[:, k] for k in range(Z.shape[1])]]
        if "interaction" in self.cfg.model_form:
            d = Z.shape[1]
            for a in range(d):
                for b in range(a + 1, d):
                    cols.append(Z[:, a] * Z[:, b])
        return np.column_stack(cols)

    def _design_score(self, Xm: np.ndarray, Mref: np.ndarray, objective: str) -> float:
        XtX = Xm.T @ Xm + 1e-6 * np.eye(Xm.shape[1])
        if objective == "D":
            sign, logdet = np.linalg.slogdet(XtX)
            return float(logdet)                       # maximize log-det (param precision)
        # I-optimality: minimize average prediction variance over reference => maximize negative
        XtX_inv = np.linalg.inv(XtX)
        pv = np.einsum("ij,jk,ik->i", Mref, XtX_inv, Mref)
        return float(-np.mean(pv))

    # -- EP-04/05 : sequential infill -----------------------------------
    def sequential_infill(self, X_existing: np.ndarray, adjustable: List[str],
                          n_new: int, acquisition: str = "alc",
                          y_existing: Optional[np.ndarray] = None) -> List[ExperimentProposal]:
        """GP-surrogate infill choosing experiments that most reduce uncertainty (EP-04)."""
        d = len(adjustable)
        if y_existing is None:
            # synthetic response in the gap: variance-seeking only
            y_existing = np.zeros(len(X_existing))
        gp = GaussianProcess(length_scale=0.6, signal_var=float(np.var(y_existing) + 1.0),
                             noise_var=1e-3)
        Xcur = np.atleast_2d(X_existing).astype(float)
        ycur = np.asarray(y_existing, dtype=float)
        gp.fit(Xcur, ycur)
        ref = self.rng.uniform(self.cfg.level_low, self.cfg.level_high, size=(300, d))
        proposals = []
        for _ in range(n_new):
            cand = self.rng.uniform(self.cfg.level_low, self.cfg.level_high,
                                    size=(self.cfg.candidate_pool, d))
            if acquisition == "eig":
                scores = gp.expected_information_gain(cand)
            elif acquisition == "imse" or acquisition == "alc":
                scores = np.array([gp.alc_score(c, ref) for c in cand])
            else:
                raise ValueError(f"unknown acquisition {acquisition!r}")
            best = int(np.argmax(scores))
            x_new = cand[best]
            info_gain = float(scores[best])
            # predicted model improvement (EP-05): reduction in integrated variance
            pv_before = float(np.mean(gp.predictive_variance(ref)))
            Xcur = np.vstack([Xcur, x_new])
            ycur = np.append(ycur, gp.predict(x_new[None, :])[0])
            gp.fit(Xcur, ycur)
            pv_after = float(np.mean(gp.predictive_variance(ref)))
            improvement = max(pv_before - pv_after, 0.0)
            # which directions it excites
            excites = [adjustable[k] for k in np.argsort(np.abs(x_new))[::-1][:2]]
            proposals.append(ExperimentProposal(
                setpoint={adjustable[k]: float(x_new[k]) for k in range(d)},
                excites_directions=excites,
                information_gain=info_gain,
                predicted_model_improvement=improvement,
                priority=info_gain * (1.0 + improvement),
            ))
        proposals.sort(key=lambda p: p.priority, reverse=True)
        return proposals

    # -- EP-07 : gate-based workflow ------------------------------------
    def plan(self, X_passive: np.ndarray, feature_names: List[str],
             adjustable: List[str], y_passive: Optional[np.ndarray] = None) -> PlannerResult:
        """framing -> diagnosis -> seed -> fit/re-diagnosis -> infill -> stopping (EP-07).

        Operates under manual approval: returns proposals for a scientist to
        approve, never executes them (EP-06).
        """
        log = ["framing: objective=%s, loop_status=%s, model_form=%s (EP-08)" %
               (self.cfg.primary_objective, self.cfg.loop_status, self.cfg.model_form)]

        gap = self.diagnose_gap(X_passive, feature_names, adjustable)
        log.append(f"passive-data diagnosis: {gap.confounding_summary}")

        # restrict the design space to adjustable inputs (EP-02)
        adj_idx = [feature_names.index(a) for a in adjustable if a in feature_names]
        Xadj = np.atleast_2d(X_passive)[:, adj_idx] if adj_idx else np.atleast_2d(X_passive)

        seed = self.seed_design(adjustable)
        log.append(f"seed design: {len(seed)} runs, {self.cfg.primary_objective}-optimal")

        # sequential infill in the gap, with stopping (EP-07)
        proposals = []
        remaining = self.cfg.budget
        stopped = "budget exhausted"
        infill = self.sequential_infill(seed, adjustable, n_new=remaining,
                                        acquisition="alc")
        kept = []
        for p in infill:
            if p.information_gain < self.cfg.stopping_info_gain:
                stopped = (f"marginal information gain {p.information_gain:.2e} below "
                           f"threshold {self.cfg.stopping_info_gain:.1e}")
                break
            kept.append(p)
        proposals = kept if kept else infill[:1]
        log.append(f"sequential infill: {len(proposals)} experiments proposed; "
                   f"stop because {stopped}")
        log.append("MANUAL APPROVAL required: proposals presented, not executed (EP-06)")

        return PlannerResult(
            gap=gap, proposals=proposals, gate_log=log,
            config={**self.cfg.__dict__}, stopped_reason=stopped,
        )
