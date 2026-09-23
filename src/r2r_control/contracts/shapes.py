"""FB-model shape contract (DM-04).

``M`` maps knob deltas to output deltas and MUST be ``n_out x n_knob``. The
contract is enforced at ingestion and at every interface that touches ``M``,
failing loudly on mismatch.
"""

from __future__ import annotations

import numpy as np


class ShapeContractError(ValueError):
    """Raised when an ``M``/dynamics artifact violates the n_out x n_knob contract."""


def check_shape_contract(M: np.ndarray, n_out: int, n_knob: int, where: str = "M") -> np.ndarray:
    """Enforce and verify the FB-model shape contract (DM-04)."""
    M = np.asarray(M, dtype=float)
    if M.shape != (n_out, n_knob):
        raise ShapeContractError(
            f"{where} has shape {M.shape}, expected ({n_out}, {n_knob}) "
            f"[n_out x n_knob]"
        )
    return M
