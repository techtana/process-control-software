"""Open-loop baseline u0: source, validity horizon, drift check (CF-02, A2/E3).

``u0`` is a single point of failure that CF, SIM, REG and DIAG all run through.
A drifting tool's open-loop operating point drifts, so a static ``u0`` is only
valid within a bounded window; beyond it the counterfactual picks up a slow
error that contaminates the decoupled regression with the very non-stationary
trend the innovation-target approach was meant to remove. This module sources
``u0``, checks its validity horizon, and can return a time-varying baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ..excitation.accidental import runs


@dataclass
class BaselineResult:
    u0: np.ndarray                       # (n_knob,) static, or (T, n_knob) time-varying
    source: str
    time_varying: bool = False
    drift_detected: bool = False
    validity_horizon: Optional[int] = None
    per_episode: List[dict] = field(default_factory=list)
    warning: str = ""

    def to_dict(self):
        return {"u0": np.asarray(self.u0).tolist(), "source": self.source,
                "time_varying": self.time_varying, "drift_detected": self.drift_detected,
                "validity_horizon": self.validity_horizon, "per_episode": self.per_episode,
                "warning": self.warning}


def estimate_baseline(used_knobs: np.ndarray, control_off_mask: Optional[np.ndarray] = None,
                      times: Optional[np.ndarray] = None, drift_span: float = 1.0,
                      allow_time_varying: bool = False) -> BaselineResult:
    """Source ``u0`` and check its validity over the horizon (CF-02, A2/E3).

    Prefers the control-off operating point (the true open-loop baseline). With
    several control-off episodes over time, their baselines are regressed on
    episode time; if the fitted change across the record exceeds ``drift_span``
    knob standard deviations, ``u0`` drift is flagged with a re-baseline
    horizon, and (if ``allow_time_varying``) an interpolated baseline returned.
    """
    U = np.asarray(used_knobs, dtype=float)
    T, n_knob = U.shape
    times = np.arange(T, dtype=float) if times is None else np.asarray(times, dtype=float)

    if control_off_mask is None or not np.any(control_off_mask):
        return BaselineResult(
            u0=np.nanmean(U, axis=0), source="global knob mean (no control-off episodes)",
            warning="u0 not anchored to a true open-loop baseline; confirm the initial "
                    "knob setting (CF-02, single point of failure)")

    episodes = runs(np.asarray(control_off_mask, dtype=bool))
    centers = np.array([np.nanmean(U[a:b + 1], axis=0) for a, b in episodes])
    ep_times = np.array([float(np.mean(times[a:b + 1])) for a, b in episodes])
    per_episode = [{"episode": ep, "time": t, "u0": c.tolist()}
                   for ep, t, c in zip(episodes, ep_times, centers)]

    drift_detected, horizon, warning = False, None, ""
    span = float(ep_times.max() - ep_times.min()) if len(episodes) >= 2 else 0.0
    if span > 0:
        sd = np.nanstd(U, axis=0)
        sd = np.where(sd > 1e-12, sd, 1.0)
        slopes = np.array([np.polyfit(ep_times, centers[:, j], 1)[0] for j in range(n_knob)])
        spreads = np.abs(slopes) * span / sd
        max_spread = float(spreads.max())
        if max_spread > drift_span:
            drift_detected = True
            horizon = int(max(span / max_spread, 1))
            warning = (f"u0 drift detected across control-off episodes (fitted change "
                       f"{max_spread:.2f} knob-std over the record); re-baseline every "
                       f"~{horizon} events (A2/E3)")

    if drift_detected and allow_time_varying:
        u0_tv = np.column_stack([np.interp(times, ep_times, centers[:, j])
                                 for j in range(n_knob)])
        return BaselineResult(u0=u0_tv, source="time-varying control-off baseline (interpolated)",
                              time_varying=True, drift_detected=True, validity_horizon=horizon,
                              per_episode=per_episode, warning=warning)
    return BaselineResult(u0=centers.mean(axis=0), source="control-off episode mean",
                          drift_detected=drift_detected, validity_horizon=horizon,
                          per_episode=per_episode, warning=warning)
