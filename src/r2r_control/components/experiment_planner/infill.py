"""EP-04 — GP-surrogate sequential infill (ALC / IMSE / EIG), discrete-aware (E11)."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ...core.gp import GaussianProcess, round_to_feasible
from .value import ExperimentProposal, priority


def sequential_infill(X_existing: np.ndarray, adjustable: List[str], n_new: int,
                      level_low: float, level_high: float, candidate_pool: int,
                      rng: np.random.Generator, acquisition: str = "alc",
                      y_existing: Optional[np.ndarray] = None, integer_dims=None,
                      discount: float = 1.0) -> List[ExperimentProposal]:
    """Pick experiments that most reduce model uncertainty in the gap (EP-04/05).

    ``discount`` (A6/E9) scales the information credited to each experiment by
    the fraction of excitation expected to survive the controller.
    """
    d = len(adjustable)
    Xcur = np.atleast_2d(X_existing).astype(float)
    ycur = np.zeros(len(Xcur)) if y_existing is None else np.asarray(y_existing, dtype=float)
    gp = GaussianProcess(length_scale=0.6, signal_var=float(np.var(ycur) + 1.0), noise_var=1e-3)
    gp.fit(Xcur, ycur)
    ref = rng.uniform(level_low, level_high, size=(300, d))
    proposals = []
    for _ in range(n_new):
        cand = rng.uniform(level_low, level_high, size=(candidate_pool, d))
        if integer_dims:
            cand = round_to_feasible(cand, integer_dims=integer_dims)   # E11
        if acquisition == "eig":
            scores = gp.expected_information_gain(cand)
        elif acquisition in ("imse", "alc"):
            scores = np.array([gp.alc_score(c, ref) for c in cand])
        else:
            raise ValueError(f"unknown acquisition {acquisition!r}")
        best = int(np.argmax(scores))
        x_new = cand[best]
        info_gain = discount * float(scores[best])
        pv_before = float(np.mean(gp.predictive_variance(ref)))
        Xcur = np.vstack([Xcur, x_new])
        ycur = np.append(ycur, gp.predict(x_new[None, :])[0])
        gp.fit(Xcur, ycur)
        improvement = discount * max(pv_before - float(np.mean(gp.predictive_variance(ref))), 0.0)
        proposals.append(ExperimentProposal(
            setpoint={adjustable[k]: float(x_new[k]) for k in range(d)},
            excites_directions=[adjustable[k] for k in np.argsort(np.abs(x_new))[::-1][:2]],
            information_gain=info_gain, predicted_model_improvement=improvement,
            priority=priority(info_gain, improvement)))
    proposals.sort(key=lambda p: p.priority, reverse=True)
    return proposals
