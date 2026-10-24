"""EP-04 — GP-surrogate sequential infill (ALC / IMSE / EIG), gap-targeted (KI-6).

The acquisition score of each candidate is multiplied by
``1 + gap_weight · alignment``, where ``alignment`` is the share of the
candidate's coded excursion (from the design center) that lies along the
identifiability-gap directions, so experiments aim at what the passive data
cannot identify. With observed responses the GP hyperparameters are chosen by
marginal likelihood; without them the surrogate is purely geometric (a
space-filling criterion), which the caller reports.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ...core.gp import GaussianProcess
from .space import CodedSpace
from .value import ExperimentProposal, priority, run_cost


def gap_alignment(X: np.ndarray, directions: np.ndarray, center: float) -> np.ndarray:
    """Share of each point's excursion from ``center`` lying in span(directions)."""
    Xc = np.atleast_2d(X) - center
    if directions.size == 0:
        return np.zeros(len(Xc))
    Q, _ = np.linalg.qr(directions.T)                 # orthonormal basis of the gap span
    proj = Xc @ Q
    nrm = np.sum(Xc ** 2, axis=1)
    return np.where(nrm > 1e-12, np.sum(proj ** 2, axis=1) / np.maximum(nrm, 1e-12), 0.0)


def sequential_infill(X_existing: np.ndarray, adjustable: List[str], n_new: int,
                      level_low: float, level_high: float, candidate_pool: int,
                      rng: np.random.Generator, acquisition: str = "alc",
                      y_existing: Optional[np.ndarray] = None, integer_dims=None,
                      discount: float = 1.0, space: Optional[CodedSpace] = None,
                      gap_directions: Optional[np.ndarray] = None, gap_weight: float = 4.0,
                      base_cost: float = 1.0, knob_cost=None,
                      purpose: str = "") -> List[ExperimentProposal]:
    """Pick experiments that most reduce model uncertainty in the gap (EP-04/05).

    ``discount`` (A6/E9) scales the information credited to each experiment by
    the fraction of excitation expected to survive the controller.
    """
    space = space or CodedSpace(list(adjustable), level_low, level_high,
                                integer_dims=integer_dims)
    d = len(adjustable)
    V = np.zeros((0, d)) if gap_directions is None else np.atleast_2d(gap_directions)
    Xcur = np.atleast_2d(X_existing).astype(float)
    have_y = y_existing is not None and len(np.ravel(y_existing)) >= 3
    gp = GaussianProcess(length_scale=0.6, signal_var=1.0, noise_var=1e-3,
                         categorical_dims=list(space.categorical_levels or {}) or None)
    if have_y:
        ycur = np.asarray(y_existing, dtype=float).reshape(-1)
        gp.fit_hyperparameters(Xcur, ycur)
    else:
        ycur = np.zeros(len(Xcur))
        gp.fit(Xcur, ycur)
    ref = space.reference(rng, 300)
    proposals = []
    for _ in range(n_new):
        cand = space.sample(rng, candidate_pool)
        if acquisition == "eig":
            scores = gp.expected_information_gain(cand)
        elif acquisition in ("imse", "alc"):
            scores = np.array([gp.alc_score(c, ref) for c in cand])
        else:
            raise ValueError(f"unknown acquisition {acquisition!r}")
        align = gap_alignment(cand, V, space.center)
        best = int(np.argmax(scores * (1.0 + gap_weight * align)))
        x_new = cand[best]
        info_gain = discount * float(scores[best])
        pv_before = float(np.mean(gp.predictive_variance(ref)))
        Xcur = np.vstack([Xcur, x_new])
        ycur = np.append(ycur, gp.predict(x_new[None, :])[0])
        gp.fit(Xcur, ycur)
        improvement = discount * max(pv_before - float(np.mean(gp.predictive_variance(ref))), 0.0)
        cost = run_cost(x_new, adjustable, space.center, base_cost, knob_cost)
        coded, eng = space.setpoints(x_new)
        proposals.append(ExperimentProposal(
            setpoint=coded,
            excites_directions=[adjustable[k] for k in
                                np.argsort(np.abs(x_new - space.center))[::-1][:2]],
            information_gain=info_gain, predicted_model_improvement=improvement,
            priority=priority(info_gain, improvement, cost), cost=cost,
            setpoint_engineering=eng, gap_alignment=float(align[best]), purpose=purpose))
    proposals.sort(key=lambda p: p.priority, reverse=True)
    return proposals
