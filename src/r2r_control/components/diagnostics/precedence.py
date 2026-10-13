"""DIAG-07 — diagnostic precedence and conditioning (E4, E5).

The diagnostics have overlapping signatures, so they run in a fixed order rather
than as a flat panel: over-aggressive correction can come from gain
underestimation, a hot Controller QR, or an over-trusting State QR (a three-way
confound). The tree is

  1. excitation gate — is there enough knob movement to see the gain at all?
  2. gain mismatch   — only if (1) passes
  3. FF leakage      — needs no knob movement
  4. QR attribution  — only if (1) passes AND gain mismatch is ruled out
  5. variance decomposition
  6. achievability verdict

and any node can answer "unidentifiable from available data — excitation
required", which routes the question to the experiment planner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .ff_leakage import ff_leakage
from .gain_mismatch import gain_mismatch, gain_drift_monitor
from .qr_attribution import qr_attribution
from .variance_decomp import variance_decomposition
from .achievability import achievability_verdict

UNIDENTIFIABLE = "unidentifiable from available data — excitation required"


@dataclass
class Diagnosis:
    excitation: Dict
    ff_leakage: Dict
    gain_mismatch: Dict
    qr_attribution: Dict
    variance_decomposition: Dict
    achievability: Dict
    gain_drift: Optional[Dict] = None
    precedence_log: List[str] = field(default_factory=list)
    routed_to_planner: List[str] = field(default_factory=list)
    figures: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self):
        return dict(self.__dict__)


def excitation_sufficiency(u_used: np.ndarray, measured_mask: Optional[np.ndarray] = None,
                           rel_floor: float = 0.75) -> Dict:
    """Node 1: is there enough knob movement to estimate realized gain? (E4)

    Under tight control Δu ≈ 0 makes the realized gain Δy/Δu undefined. We
    compare the RMS per-event knob move with the knob's own operating spread (a
    scale-free ratio): a rate-limited knob only creeps, so consecutive events
    carry little independent excitation however healthy the loop looks.

    Calibration (synthetic MIMO loops, 5 seeds): tight/rate-limited loops score
    0.25-0.59, actively correcting healthy loops 0.95-2.7, over-correcting loops
    2.0-4.0; the default floor sits in the gap.
    """
    U = np.asarray(u_used, dtype=float)
    if measured_mask is not None:
        U = U[np.where(measured_mask)[0]]
    du = np.diff(U, axis=0)
    du = du[np.all(np.isfinite(du), axis=1)]
    if len(du) < du.shape[1] + 1:
        return {"sufficient": False, "move_ratio": 0.0, "move_energy": 0.0,
                "reason": "too few knob moves to estimate realized gain (E4)"}
    move_energy = float(np.mean(np.sum(du ** 2, axis=1)))
    spread = np.nanstd(U, axis=0)
    scale = float(np.mean(spread[spread > 0])) if np.any(spread > 0) else 1.0
    move_ratio = np.sqrt(move_energy) / (scale + 1e-12)
    sufficient = move_ratio > rel_floor
    return {"sufficient": bool(sufficient), "move_ratio": float(move_ratio),
            "move_energy": move_energy,
            "reason": ("adequate knob movement to estimate realized gain" if sufficient
                       else "Δu≈0 under tight control: realized gain undefined (E4)")}


class DiagnosticsEngine:
    """Component 5. Attributes residual variance via the DIAG-07 decision tree."""

    def __init__(self, delay: int = 1, alpha: float = 0.05, excitation_floor: float = 0.75):
        self.delay = delay
        self.alpha = alpha
        self.excitation_floor = excitation_floor

    # leaf diagnostics stay callable directly (monitoring / ad-hoc use)
    ff_leakage = staticmethod(ff_leakage)
    gain_mismatch = staticmethod(gain_mismatch)
    qr_attribution = staticmethod(qr_attribution)
    gain_drift_monitor = staticmethod(gain_drift_monitor)
    achievability_verdict = staticmethod(achievability_verdict)

    def diagnose(self, y_observed, u_used, M_model, state_est=None, ff_inputs=None,
                 innovation=None, ff_names=None, measured_mask=None, figure_dir=None,
                 monitor_gain_drift: bool = False) -> Diagnosis:
        log: List[str] = []
        routed: List[str] = []

        # (1) excitation gate
        exc = excitation_sufficiency(u_used, measured_mask, self.excitation_floor)
        log.append(f"1. excitation: {'sufficient' if exc['sufficient'] else 'INSUFFICIENT'} "
                   f"({exc['reason']})")

        # (2) gain mismatch
        if exc["sufficient"]:
            gm = gain_mismatch(y_observed, u_used, M_model, innovation, measured_mask)
            log.append(f"2. gain mismatch: {gm['diagnosis']}")
        else:
            gm = {"diagnosis": UNIDENTIFIABLE, "gain_underestimated": None,
                  "oscillating": None}
            routed.append("process gain (excitation required)")
            log.append(f"2. gain mismatch: {UNIDENTIFIABLE}")

        # (3) FF leakage
        if state_est is not None and ff_inputs is not None:
            ff = ff_leakage(state_est, ff_inputs, ff_names, self.alpha)
        else:
            ff = {"leaking_inputs": [], "feedforward_opportunities": [],
                  "detail": "no state estimate / FF inputs provided"}
        log.append(f"3. FF leakage: {ff['feedforward_opportunities']}")

        # (4) QR attribution
        if not exc["sufficient"]:
            qr = {"primary_suspect": UNIDENTIFIABLE, "status": UNIDENTIFIABLE}
            routed.append("Controller-QR vs State-QR (excitation required)")
            log.append(f"4. QR attribution: {UNIDENTIFIABLE}")
        elif gm.get("gain_underestimated"):
            qr = {"primary_suspect": "deferred",
                  "status": "deferred: resolve gain mismatch first (DIAG-07/E5)",
                  "reason": "QR effects are confounded with the gain error (FB-08)"}
            log.append("4. QR attribution: deferred until gain mismatch is resolved (E5)")
        else:
            qr = qr_attribution(innovation if innovation is not None
                                else np.diff(np.asarray(y_observed, dtype=float), axis=0),
                                u_used, y_observed, self.alpha)
            log.append(f"4. QR attribution: {qr['primary_suspect']}")

        # (5) variance decomposition, (6) achievability
        vd = variance_decomposition(y_observed, ff, gm, qr, self.delay)
        log.append("5. variance decomposition computed")
        ach = achievability_verdict(y_observed, self.delay)
        log.append(f"6. achievability: {'at floor' if ach['at_achievability_floor'] else 'gap'}")

        gd = gain_drift_monitor(y_observed, u_used, measured_mask=measured_mask) \
            if monitor_gain_drift else None

        notes = [ach["verdict"]]
        if ff["leaking_inputs"]:
            notes.append(f"FF leakage: {ff['leaking_inputs']} are feedforward opportunities")
        if gm["diagnosis"] not in ("none", UNIDENTIFIABLE):
            notes.append(f"gain/stability: {gm['diagnosis']}")
        if routed:
            notes.append(f"routed to experiment planner: {routed}")
        if gd and gd["drift_detected"]:
            notes.append("gain drift detected — re-identify (A1/E1)")

        figures = []
        if figure_dir is not None:
            from .viz import visualize
            figures = visualize(y_observed, u_used, M_model, innovation, ach, vd,
                                figure_dir, measured_mask)
        return Diagnosis(excitation=exc, ff_leakage=ff, gain_mismatch=gm, qr_attribution=qr,
                         variance_decomposition=vd, achievability=ach, gain_drift=gd,
                         precedence_log=log, routed_to_planner=routed, figures=figures,
                         notes=notes)
