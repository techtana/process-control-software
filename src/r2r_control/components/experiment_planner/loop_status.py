"""EP-08 / A6 / E9 — closed-loop excitation discount.

Information calculations assume the designed excitation reaches the process. If
the loop stays closed during the experiment, the controller treats the
perturbation as a disturbance and partly cancels it, so the experiment delivers
less information than the optimality calculation credits — closed-loop
confounding reappearing inside experiment design. Guards: open the loop, inject
as a pass-through setpoint/dither, or compute information against the
post-control *residual* excitation.
"""

from __future__ import annotations

import numpy as np


def closed_loop_excitation_discount(loop_status: str, rejection_fraction: float = 0.6) -> float:
    """Fraction of commanded excitation that lands: 1 if open, 1-rejection if closed."""
    if loop_status == "open":
        return 1.0
    if loop_status != "closed":
        raise ValueError(f"loop_status must be 'open' or 'closed', got {loop_status!r}")
    return float(np.clip(1.0 - rejection_fraction, 0.0, 1.0))


def residual_excitation(commanded: np.ndarray, applied_residual: np.ndarray) -> float:
    """Measured surviving fraction of commanded excitation variance (A6/E9)."""
    c = float(np.var(np.asarray(commanded, dtype=float)))
    if c <= 1e-12:
        return 0.0
    return float(np.clip(np.var(np.asarray(applied_residual, dtype=float)) / c, 0.0, 1.0))
