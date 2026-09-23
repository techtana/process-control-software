"""REG-02 — controller decoupling with the phantom-FF guard (A3/E2).

With a controller in place, relationships are estimated against the
open-loop-equivalent signal via the counterfactual subtraction ``M (u_used -
u0)``. That ``M`` is only accurate to within its uncertainty (A3): structured
error in ``M`` leaves a structured artifact in the decoupled residual, and any
sensor correlated with the directions ``M`` got wrong will seem to "explain" it —
a **phantom feedforward candidate** that is really explaining model error. This
module propagates ``M`` uncertainty into the target's effective noise and
quarantines sensors whose significant apparent effect is no larger than the
``M``-error envelope along the knob directions they load on (E2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ...core.counterfactual.reconstruct import reconstruct


@dataclass
class DecoupleResult:
    target: np.ndarray
    decoupled: bool
    effective_noise_inflation: float = 0.0
    quarantined_sensors: List[str] = field(default_factory=list)
    phantom_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)


def phantom_ff_scores(ff_matrix: np.ndarray, ff_names: List[str], knob_deltas: np.ndarray,
                      target: np.ndarray, m_row: np.ndarray, rel_uncertainty_M: float,
                      n_sigma: float = 2.0) -> Dict[str, Dict[str, float]]:
    """Compare each sensor's apparent effect with what an ``M`` error could fake (A3/E2).

    For sensor ``f_j``: ``b_j`` is its slope on the decoupled target and ``a_j``
    how strongly the knob deltas load on it. A relative ``M`` error of size
    ``rel_uncertainty_M`` leaves ``δm · Δu`` in the target, which projects onto
    ``f_j`` with a slope of at most about ``rel_uncertainty_M·‖m_row‖·‖a_j‖``.
    A sensor whose effect is significant yet inside ``n_sigma`` times that
    envelope cannot be told apart from model error.
    """
    K = np.asarray(knob_deltas, dtype=float)
    ok = np.isfinite(target)
    Kc = K[ok] - K[ok].mean(0)
    t = target[ok] - target[ok].mean()
    n = int(ok.sum())
    scores = {}
    for j, nm in enumerate(ff_names):
        f = np.asarray(ff_matrix[ok, j], dtype=float)
        fc = f - f.mean()
        ss = float(fc @ fc)
        if ss < 1e-12 or n < 5:
            scores[nm] = {"slope": 0.0, "t_stat": 0.0, "envelope": 0.0, "r2_on_knobs": 0.0}
            continue
        a = (fc @ Kc) / ss                          # knob loading on the sensor
        b = float(fc @ t) / ss                      # apparent effect on the target
        resid = t - b * fc
        se = np.sqrt(float(resid @ resid) / max(n - 2, 1) / ss)
        fit = Kc @ np.linalg.lstsq(Kc, fc, rcond=None)[0]
        scores[nm] = {"slope": b, "t_stat": b / (se + 1e-12),
                      "envelope": n_sigma * rel_uncertainty_M * float(np.linalg.norm(m_row))
                      * float(np.linalg.norm(a)),
                      "r2_on_knobs": float(1.0 - np.sum((fc - fit) ** 2) / ss)}
    return scores


def decouple(Y: np.ndarray, M: Optional[np.ndarray], u_used: np.ndarray,
             u0: Optional[np.ndarray], j_out: int, ff_matrix: Optional[np.ndarray] = None,
             ff_names: Optional[List[str]] = None, rel_uncertainty_M: float = 0.0,
             provenance=None) -> DecoupleResult:
    """Subtract the control contribution and guard against phantom FF (REG-02, A3/E2).

    Returns ``decoupled=False`` when ``M`` or ``u0`` is missing; the caller then
    falls back to the innovation target.
    """
    if M is None or u0 is None:
        return DecoupleResult(target=Y[:, j_out], decoupled=False)
    cf = reconstruct(np.nan_to_num(Y), M, u_used, u0)
    target = np.where(np.isfinite(Y[:, j_out]), cf.y_nocontrol[:, j_out], np.nan)
    if provenance is not None:
        provenance.note("decoupled control contribution via counterfactual subtraction "
                        "M@(u_used-u0) (REG-02)")

    u0 = np.asarray(u0, dtype=float)
    knob_deltas = u_used - (u0 if u0.ndim == 2 else u0[None, :])
    inflation = float(rel_uncertainty_M * np.linalg.norm(M) *
                      np.sqrt(np.mean(np.sum(knob_deltas ** 2, axis=1))))
    scores, quarantined = {}, []
    if ff_matrix is not None and ff_names and rel_uncertainty_M > 0:
        scores = phantom_ff_scores(ff_matrix, ff_names, knob_deltas, target,
                                   np.asarray(M)[j_out], rel_uncertainty_M)
        quarantined = [nm for nm, s in scores.items()
                       if abs(s["t_stat"]) > 2.0 and abs(s["slope"]) <= s["envelope"]]
        if provenance is not None:
            provenance.note(f"M uncertainty {rel_uncertainty_M:.2f} inflates target noise "
                            f"by ~{inflation:.3f} (A3)")
            if quarantined:
                provenance.note(f"phantom-FF guard quarantined {quarantined}: they align "
                                f"with the knob/M-error directions, so they likely explain "
                                f"model error rather than a real disturbance (A3/E2)")
    return DecoupleResult(target=target, decoupled=True, effective_noise_inflation=inflation,
                          quarantined_sensors=quarantined, phantom_scores=scores)
