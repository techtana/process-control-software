"""Subspace-style model-order selection (FB-02) — Component 1 only.

Model order = the number of singular values above the noise gap.
"""

from __future__ import annotations

import numpy as np

from ...core.excitation.spectrum import numerical_rank


def subspace_order(X: np.ndarray, rtol: float = 1e-2) -> int:
    """Order = singular values of the (centered) regressor block above the gap."""
    Xc = np.asarray(X, dtype=float)
    Xc = Xc - Xc.mean(0)
    return numerical_rank(np.linalg.svd(Xc, compute_uv=False), rtol)
