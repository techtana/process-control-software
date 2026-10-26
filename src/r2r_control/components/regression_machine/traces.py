"""REG-09 — representation choice for traces (physics-aware aggregation).

Traces are array-valued cells of differing length (the ragged-panel problem).
The default representation is per-substep aggregation (mean / std per substep,
plus overall mean, std and ramp rate), which injects prior and cuts dimension.
The raw trace (resampled to a common length) is retained only when a held-out
check shows aggregation destroys discriminative signal, e.g. a transient that
the mean hides.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np


def aggregate_trace(trace: np.ndarray, substeps: int = 1) -> Dict[str, float]:
    """Per-substep aggregation of a single trace (REG-09).

    Aggregation is the default: it injects prior knowledge and cuts dimension
    before fitting. Ramp rate is exposed alongside the mean so the "does
    aggregation destroy signal?" test can be made.
    """
    t = np.asarray(trace, dtype=float) if trace is not None else np.array([])
    t = t[np.isfinite(t)] if t.size else t
    keys = [f"sub{i}_{k}" for i in range(max(substeps, 1)) for k in ("mean", "std")]
    if t.size == 0:
        return {**{k: np.nan for k in keys}, "mean": np.nan, "std": np.nan,
                "ramp_rate": np.nan}
    feats: Dict[str, float] = {}
    for i, ch in enumerate(np.array_split(t, max(substeps, 1))):
        feats[f"sub{i}_mean"] = float(np.mean(ch)) if ch.size else np.nan
        feats[f"sub{i}_std"] = float(np.std(ch)) if ch.size else np.nan
    feats["mean"] = float(np.mean(t))
    feats["std"] = float(np.std(t))
    feats["ramp_rate"] = float((t[-1] - t[0]) / max(len(t) - 1, 1))
    return feats


def aggregation_destroys_signal(raw_corr: float, agg_corr: float, margin: float = 0.1) -> bool:
    """True only if the raw trace predicts the target materially better than the
    aggregate — the one case where the raw trace is retained (REG-09)."""
    return raw_corr - agg_corr > margin


def trace_features(cells: Sequence, name: str, substeps: int = 1
                   ) -> Tuple[np.ndarray, List[str]]:
    """Aggregated features for one trace column: matrix (T x k) and names."""
    rows = [aggregate_trace(c, substeps) for c in cells]
    keys = list(rows[0]) if rows else []
    F = np.array([[r[k] for k in keys] for r in rows], dtype=float)
    return F, [f"{name}_{k}" for k in keys]


def resample_traces(cells: Sequence, length: int = 16) -> np.ndarray:
    """Ragged traces linearly resampled to a common length (T x length)."""
    out = np.full((len(cells), length), np.nan)
    grid = np.linspace(0.0, 1.0, length)
    for i, c in enumerate(cells):
        t = np.asarray(c, dtype=float) if c is not None else np.array([])
        t = t[np.isfinite(t)] if t.size else t
        if t.size >= 2:
            out[i] = np.interp(grid, np.linspace(0.0, 1.0, t.size), t)
        elif t.size == 1:
            out[i] = t[0]
    return out


def raw_retention_check(raw: np.ndarray, agg: np.ndarray, target: np.ndarray,
                        margin: float = 0.1) -> Dict[str, float]:
    """Held-out comparison of raw vs aggregated trace representations (REG-09).

    Rows are split in time: on the first half pick the best single aggregated
    feature and fit a one-component PLS direction on the raw trace; on the
    second half compare their |correlation| with the target.
    """
    ok = np.isfinite(target) & np.all(np.isfinite(raw), axis=1) & np.all(np.isfinite(agg), axis=1)
    idx = np.where(ok)[0]
    if len(idx) < 20:
        return {"raw_corr": np.nan, "agg_corr": np.nan, "retain_raw": False,
                "detail": "too few complete rows to compare representations"}
    tr, te = idx[:len(idx) // 2], idx[len(idx) // 2:]

    def corr(a, b):
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return 0.0
        return float(abs(np.corrcoef(a, b)[0, 1]))

    best_j = int(np.argmax([corr(agg[tr, j], target[tr]) for j in range(agg.shape[1])]))
    agg_corr = corr(agg[te, best_j], target[te])
    Rc = raw[tr] - raw[tr].mean(0)
    w = Rc.T @ (target[tr] - target[tr].mean())
    w = w / (np.linalg.norm(w) + 1e-12)
    raw_corr = corr((raw[te] - raw[tr].mean(0)) @ w, target[te])
    retain = aggregation_destroys_signal(raw_corr, agg_corr, margin)
    return {"raw_corr": raw_corr, "agg_corr": agg_corr, "retain_raw": bool(retain),
            "detail": ("raw trace predicts the target materially better than any aggregate "
                       "=> raw (resampled) trace retained" if retain
                       else "aggregation preserves the signal => aggregated features only")}
