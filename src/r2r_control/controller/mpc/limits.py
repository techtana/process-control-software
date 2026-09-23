"""Move / rate limits (SIM-04)."""

from __future__ import annotations

import numpy as np


def apply_move_limit(u_cmd: np.ndarray, u_prev: np.ndarray, move_limit: float) -> np.ndarray:
    """Clip the per-event knob move to +/- ``move_limit`` (rate limit, SIM-04)."""
    return u_prev + np.clip(u_cmd - u_prev, -move_limit, move_limit)
