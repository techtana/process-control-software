"""DIAG-01 — FF-leakage diagnostic."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from scipy.stats import norm


def ff_leakage(state_est: np.ndarray, ff_inputs: np.ndarray,
               ff_names: Optional[List[str]] = None, alpha: float = 0.05) -> Dict:
    """Test whether changes in estimated state correlate with FF inputs (DIAG-01).

    A significant correlation means the FF model misses a measurable
    disturbance, so feedback is reactively absorbing what feedforward should
    pre-compensate. Those inputs are reported as feedforward opportunities.
    """
    state_est = np.asarray(state_est, dtype=float)
    ff = np.asarray(ff_inputs, dtype=float)
    if ff.size == 0 or len(state_est) < 5:
        return {"leaking_inputs": [], "feedforward_opportunities": [],
                "detail": "no FF inputs / too few samples"}
    d_state = np.diff(state_est, axis=0)
    d_ff = np.diff(ff, axis=0)
    names = ff_names or [f"ff{j}" for j in range(ff.shape[1])]
    n = len(d_state)
    n_out = d_state.shape[1]
    raw_p, best_r = [], []
    for j in range(ff.shape[1]):
        best = 0.0
        for k in range(n_out):
            if np.std(d_ff[:, j]) < 1e-9 or np.std(d_state[:, k]) < 1e-9:
                continue
            r = np.corrcoef(d_ff[:, j], d_state[:, k])[0, 1]
            if abs(r) > abs(best):
                best = r
        if n > 4 and abs(best) < 1:
            z = 0.5 * np.log((1 + best) / (1 - best)) * np.sqrt(n - 3)
            p_one = 2 * norm.sf(abs(z))
            p = 1.0 - (1.0 - p_one) ** n_out      # Šidák: best of n_out correlations
        else:
            p = 1.0
        raw_p.append(float(p))
        best_r.append(float(abs(best)))
    # Holm step-down across FF inputs (family-wise error rate alpha, KI-8)
    m = len(raw_p)
    order = np.argsort(raw_p)
    adj = np.ones(m)
    running = 0.0
    for rank, j in enumerate(order):
        running = max(running, min(1.0, (m - rank) * raw_p[j]))
        adj[j] = running
    leaking, corrs = [], {}
    for j in range(m):
        corrs[names[j]] = {"max_abs_corr": best_r[j], "pvalue": raw_p[j],
                           "pvalue_holm": float(adj[j])}
        if adj[j] < alpha and best_r[j] > 0.2:
            leaking.append(names[j])
    return {"leaking_inputs": leaking, "correlations": corrs,
            "feedforward_opportunities": leaking,
            "detail": ("estimated-state vs FF correlation (Šidák over outputs, Holm over "
                       "inputs); significant => FF leakage")}
