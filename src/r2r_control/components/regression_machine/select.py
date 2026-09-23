"""REG-06/07 — low-variation detection, planner hand-off, feedforward candidates."""

from __future__ import annotations

from typing import List, Set, Tuple

import numpy as np


def low_variation(X: np.ndarray, names: List[str], adjustable: Set[str], quantile: float,
                  provenance=None) -> Tuple[List[str], List[str]]:
    """Split low-variation inputs into adjustable (=> planner) and observed-only (REG-07)."""
    if len(X) < 3:
        return [], []
    sd = np.nanstd(X, axis=0)
    scale = np.nanmedian(sd[sd > 0]) if np.any(sd > 0) else 1.0
    low_adj, obs_only = [], []
    for j, nm in enumerate(names):
        if sd[j] <= quantile * scale:
            (low_adj if nm in adjustable else obs_only).append(nm)
    if provenance is not None:
        if low_adj:
            provenance.note(f"low-variation ADJUSTABLE inputs {low_adj} => route to "
                            f"experiment planner (REG-07)")
        if obs_only:
            provenance.note(f"low-variation NON-ADJUSTABLE sensors {obs_only} => "
                            f"observed-only, unidentifiable-by-data (REG-07)")
    return low_adj, obs_only


def feedforward_candidates(importance, ff_cols: List[str], threshold: float,
                           quarantined: List[str]) -> List[str]:
    """Stably selected non-knob sensors, minus phantom-FF quarantines (REG-04, A3/E2)."""
    return [nm for nm, freq, _ in importance.ranked()
            if freq >= threshold and nm in ff_cols and nm not in quarantined]
