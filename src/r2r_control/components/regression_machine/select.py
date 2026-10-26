"""REG-06/07 — low-variation detection, planner hand-off, feedforward candidates."""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.stats import norm


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


def marginal_pvalues(X: np.ndarray, y: np.ndarray, names: List[str]) -> Dict[str, float]:
    """Two-sided Fisher-z p-value of each column's correlation with the target."""
    n = len(y)
    out = {}
    for j, nm in enumerate(names):
        x = X[:, j]
        if n < 5 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
            out[nm] = 1.0
            continue
        r = float(np.clip(np.corrcoef(x, y)[0, 1], -0.999999, 0.999999))
        out[nm] = float(2 * norm.sf(abs(np.arctanh(r)) * np.sqrt(n - 3)))
    return out


def feedforward_candidates(importance, ff_cols: List[str], threshold: float,
                           quarantined: List[str],
                           pvalues: Optional[Dict[str, float]] = None,
                           alpha: float = 0.05) -> List[str]:
    """Stably selected non-knob sensors, minus phantom-FF quarantines (REG-04, A3/E2).

    Selection frequency at a fixed penalty has no false-discovery control: with
    many null sensors some clear the threshold by chance. With ``pvalues``, a
    candidate must also be significant after a Bonferroni correction across the
    sensors screened.
    """
    bonferroni = alpha / max(len(ff_cols), 1)
    return [nm for nm, freq, _ in importance.ranked()
            if freq >= threshold and nm in ff_cols and nm not in quarantined
            and (pvalues is None or pvalues.get(nm, 1.0) < bonferroni)]
