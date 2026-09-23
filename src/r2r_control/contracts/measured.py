"""Measurement-presence predicate (DM-03 / E14).

``pandas.notna`` returns ``True`` for an array that contains only NaNs, which
silently corrupts every measured-sample count. ``_is_measured`` is the one place
the system decides whether a (possibly array-valued) cell carries a real
measurement, and it MUST gate every measurement-presence decision.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _is_measured(cell) -> bool:
    """The single measurement-presence predicate (DM-03)."""
    if cell is None:
        return False
    if isinstance(cell, float) and np.isnan(cell):
        return False
    arr = np.asarray(cell, dtype=float) if not np.isscalar(cell) else np.asarray([cell], dtype=float)
    if arr.size == 0:
        return False
    return bool(np.any(np.isfinite(arr)))


def is_measured(series_or_cell):
    """Vectorized form over a Series, or scalar form."""
    if isinstance(series_or_cell, pd.Series):
        return series_or_cell.apply(_is_measured)
    return _is_measured(series_or_cell)
