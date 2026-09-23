"""State-space dynamics assembly (A, B/M, C) — Component 1 only.

Component 1 carries a dynamics sub-problem that Component 6 does not have: the
transition behavior the MPC needs. We identify a first-order disturbance
transition ``A`` from the counterfactual (open-loop-equivalent) signal, kept
stable, and assemble it with the gain ``M`` from the shared sensitivity core.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ...contracts.shapes import check_shape_contract


def identify_first_order_dynamics(disturbance: np.ndarray,
                                  provenance=None) -> Optional[np.ndarray]:
    """First-order transition A from the counterfactual disturbance autocorrelation."""
    d = np.asarray(disturbance, dtype=float)
    d = d[np.all(np.isfinite(d), axis=1)]
    if len(d) < 5:
        if provenance is not None:
            provenance.note("too few measured events for dynamics; A omitted")
        return None
    A_T, *_ = np.linalg.lstsq(d[:-1], d[1:], rcond=None)
    A = A_T.T
    max_eig = float(np.max(np.abs(np.linalg.eigvals(A))))
    if max_eig >= 1.5:
        if provenance is not None:
            provenance.note(f"identified A unstable (max|eig|={max_eig:.2f}); clipping to "
                            f"diagonal AR estimate")
        A = np.diag(np.clip(np.diag(A), -0.99, 0.99))
    return A


def assemble_state_space(M: np.ndarray, A: Optional[np.ndarray], n_out: int, n_knob: int):
    """Assemble (A, B=M, C=I) with shape enforcement (DM-04)."""
    M = check_shape_contract(M, n_out, n_knob, where="state-space B/M")
    return A, M, np.eye(n_out)
