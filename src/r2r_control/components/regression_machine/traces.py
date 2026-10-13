"""REG-09 — representation choice for traces (physics-aware aggregation)."""

from __future__ import annotations

from typing import Dict

import numpy as np


def aggregate_trace(trace: np.ndarray, substeps: int = 1) -> Dict[str, float]:
    """Per-substep aggregation of a single trace (REG-09).

    Aggregation is the default: it injects prior knowledge and cuts dimension
    before fitting. Ramp rate is exposed alongside the mean so the "does
    aggregation destroy signal?" test can be made.
    """
    t = np.asarray(trace, dtype=float)
    t = t[np.isfinite(t)]
    if t.size == 0:
        return {"mean": np.nan, "std": np.nan, "ramp_rate": np.nan}
    feats: Dict[str, float] = {}
    for i, ch in enumerate(np.array_split(t, max(substeps, 1))):
        feats[f"sub{i}_mean"] = float(np.mean(ch))
        feats[f"sub{i}_std"] = float(np.std(ch))
    feats["mean"] = float(np.mean(t))
    feats["std"] = float(np.std(t))
    feats["ramp_rate"] = float((t[-1] - t[0]) / max(len(t) - 1, 1))
    return feats


def aggregation_destroys_signal(raw_corr: float, agg_corr: float, margin: float = 0.1) -> bool:
    """True only if the raw trace predicts the target materially better than the
    aggregate — the one case where the raw trace is retained (REG-09)."""
    return raw_corr - agg_corr > margin
