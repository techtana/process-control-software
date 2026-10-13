"""Optional eigenvalue cleaning for identification covariances (NF-05).

Nonlinear shrinkage is itself derived under the balanced-panel assumption;
prefer cross-validated cleaning strength (§VAL) over the analytic form when
panels are unbalanced.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .analytic import marchenko_pastur_edge


def eigenvalue_clip(cov: np.ndarray, floor: float) -> np.ndarray:
    """Clip eigenvalues below ``floor`` up to the floor (NF-05; SVD-truncation analog)."""
    w, V = np.linalg.eigh(np.asarray(cov, dtype=float))
    return (V * np.where(w < floor, floor, w)) @ V.T


def ledoit_peche_shrinkage(cov: np.ndarray, q: float, alpha: Optional[float] = None) -> np.ndarray:
    """Nonlinear (Ledoit-Peche-style) shrinkage toward the bulk (NF-05)."""
    cov = np.asarray(cov, dtype=float)
    w, V = np.linalg.eigh(cov)
    if alpha is not None:
        w_shrunk = (1 - alpha) * w + alpha * float(np.mean(w))
        return (V * w_shrunk) @ V.T
    _, upper = marchenko_pastur_edge(q)
    med = np.median(w[w > 0]) if np.any(w > 0) else 1.0
    bulk = w[w <= upper * med]
    target = float(np.mean(bulk)) if bulk.size else float(np.mean(w))
    weight = upper / (upper + np.clip(w, 1e-12, None))
    return (V * ((1 - weight) * w + weight * target)) @ V.T
