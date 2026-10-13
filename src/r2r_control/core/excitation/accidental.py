"""Mine accidental excitation: manual overrides and control-off episodes (EX-04)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class AccidentalExcitation:
    override_mask: np.ndarray
    control_off_mask: np.ndarray
    override_episodes: List[tuple]
    control_off_episodes: List[tuple]
    injected_variance: dict

    def to_dict(self):
        return {
            "n_override_events": int(self.override_mask.sum()),
            "n_control_off_events": int(self.control_off_mask.sum()),
            "override_episodes": self.override_episodes,
            "control_off_episodes": self.control_off_episodes,
            "injected_variance": self.injected_variance,
        }


def runs(mask: np.ndarray) -> List[tuple]:
    """Contiguous True runs of a boolean mask, as inclusive (start, end) pairs."""
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out


def mine_accidental_excitation(recommended: np.ndarray, used: np.ndarray,
                               control_off: Optional[np.ndarray] = None,
                               knob_names: Optional[List[str]] = None,
                               rel_tol: float = 1e-6) -> AccidentalExcitation:
    """Detect and catalogue 'free' excitation in otherwise deficient records (EX-04)."""
    recommended = np.asarray(recommended, dtype=float)
    used = np.asarray(used, dtype=float)
    diff = used - recommended
    scale = np.maximum(np.nanstd(used, axis=0), 1e-9)
    override_event = np.any(np.abs(diff) > rel_tol * (1.0 + scale), axis=1)
    if control_off is None:
        control_off = np.zeros(len(used), dtype=bool)
    control_off = np.asarray(control_off, dtype=bool)
    names = knob_names or [f"u{j}" for j in range(used.shape[1])]
    if override_event.any():
        injected = {nm: float(np.var(diff[override_event, j])) for j, nm in enumerate(names)}
    else:
        injected = {nm: 0.0 for nm in names}
    return AccidentalExcitation(
        override_mask=override_event, control_off_mask=control_off,
        override_episodes=runs(override_event), control_off_episodes=runs(control_off),
        injected_variance=injected,
    )
