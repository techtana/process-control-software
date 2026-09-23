"""EP-03 — D-optimal / I-optimal seed designs via coordinate exchange."""

from __future__ import annotations

import numpy as np


def model_matrix(Z: np.ndarray, model_form: str) -> np.ndarray:
    Z = np.atleast_2d(Z)
    cols = [np.ones(len(Z)), *[Z[:, k] for k in range(Z.shape[1])]]
    if "interaction" in model_form:
        d = Z.shape[1]
        cols += [Z[:, a] * Z[:, b] for a in range(d) for b in range(a + 1, d)]
    return np.column_stack(cols)


def design_score(Xm: np.ndarray, Mref: np.ndarray, objective: str) -> float:
    XtX = Xm.T @ Xm + 1e-6 * np.eye(Xm.shape[1])
    if objective == "D":
        return float(np.linalg.slogdet(XtX)[1])
    XtX_inv = np.linalg.inv(XtX)
    return float(-np.mean(np.einsum("ij,jk,ik->i", Mref, XtX_inv, Mref)))


def seed_design(n_factors: int, objective: str, model_form: str, n_runs: int,
                level_low: float, level_high: float, candidate_pool: int,
                rng: np.random.Generator, integer_dims=None) -> np.ndarray:
    """D- or I-optimal seed design via coordinate exchange (EP-03)."""
    from ...core.gp import round_to_feasible
    pool = rng.uniform(level_low, level_high, size=(candidate_pool, n_factors))
    if integer_dims:
        pool = round_to_feasible(pool, integer_dims=integer_dims)   # E11
    design = pool[rng.choice(len(pool), size=n_runs, replace=False)].copy()
    Mref = model_matrix(rng.uniform(level_low, level_high, size=(200, n_factors)), model_form)
    for _ in range(5):
        improved = False
        for i in range(n_runs):
            best_score = design_score(model_matrix(design, model_form), Mref, objective)
            best_j = None
            for j in range(len(pool)):
                trial = design.copy()
                trial[i] = pool[j]
                sc = design_score(model_matrix(trial, model_form), Mref, objective)
                if sc > best_score + 1e-9:
                    best_score, best_j = sc, j
            if best_j is not None:
                design[i] = pool[best_j]
                improved = True
        if not improved:
            break
    return design
