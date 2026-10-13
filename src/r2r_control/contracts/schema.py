"""Unified event-indexed data model (§4.1, DM-01..05).

The whole system speaks one table: an event-indexed record where each processing
event carries a timestamp, the *recommended* and *used* knob vectors, the
feedforward inputs, and — when available — the post-processing measurement(s)
with their own measurement timestamp. Metrology is sparse and asynchronous,
cells may be array-valued, and the recommended-vs-used distinction is
first-class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .measured import _is_measured
from .shapes import ShapeContractError


@dataclass
class EventTable:
    """The unified event-indexed table (DM-01)."""

    frame: pd.DataFrame
    knob_cols_recommended: List[str]
    knob_cols_used: List[str]
    ff_cols: List[str] = field(default_factory=list)
    y_cols: List[str] = field(default_factory=list)
    y_time_cols: Dict[str, str] = field(default_factory=dict)
    time_col: str = "timestamp"
    control_off_col: Optional[str] = None

    def __post_init__(self) -> None:
        if len(self.knob_cols_recommended) != len(self.knob_cols_used):
            raise ShapeContractError(
                "recommended and used knob columns must align one-to-one (DM-05)"
            )
        missing = [c for c in self._all_required_cols() if c not in self.frame.columns]
        if missing:
            raise ShapeContractError(f"EventTable frame missing columns: {missing}")

    def _all_required_cols(self) -> List[str]:
        cols = [self.time_col, *self.knob_cols_recommended, *self.knob_cols_used,
                *self.ff_cols, *self.y_cols]
        if self.control_off_col:
            cols.append(self.control_off_col)
        return cols

    @property
    def n_events(self) -> int:
        return len(self.frame)

    @property
    def n_knob(self) -> int:
        return len(self.knob_cols_used)

    @property
    def n_out(self) -> int:
        return len(self.y_cols)

    def used_knobs(self) -> np.ndarray:
        return self.frame[self.knob_cols_used].to_numpy(dtype=float)

    def recommended_knobs(self) -> np.ndarray:
        return self.frame[self.knob_cols_recommended].to_numpy(dtype=float)

    def ff(self) -> np.ndarray:
        if not self.ff_cols:
            return np.empty((self.n_events, 0))
        return self.frame[self.ff_cols].to_numpy(dtype=float)

    def regressors(self) -> np.ndarray:
        """Knob+FF regressor block used by the excitation/confounding analysis."""
        return np.column_stack([self.used_knobs(), self.ff()])

    @property
    def regressor_names(self) -> List[str]:
        return [*self.knob_cols_used, *self.ff_cols]

    def recommended_used_divergence(self) -> np.ndarray:
        """Per-event, per-knob divergence used as a first-class signal (DM-05)."""
        return self.used_knobs() - self.recommended_knobs()

    def measured_mask(self, y_col: Optional[str] = None) -> np.ndarray:
        """Boolean presence mask, gated by :func:`_is_measured` (DM-03)."""
        cols = [y_col] if y_col is not None else self.y_cols
        masks = [self.frame[c].apply(_is_measured).to_numpy() for c in cols]
        if len(masks) == 1:
            return masks[0]
        return np.column_stack(masks)

    def effective_measured_count(self, y_col: Optional[str] = None) -> int:
        """Effective measured-sample count after ``_is_measured`` filtering (DQ-05)."""
        return int(np.sum(self.measured_mask(y_col)))

    def y_matrix(self, reduce: str = "mean") -> np.ndarray:
        """Reduce (possibly array-valued) measurement cells to a numeric matrix."""
        out = np.full((self.n_events, self.n_out), np.nan)
        for j, c in enumerate(self.y_cols):
            for i, cell in enumerate(self.frame[c].to_numpy()):
                if not _is_measured(cell):
                    continue
                arr = np.asarray(cell, dtype=float)
                arr = arr[np.isfinite(arr)]
                out[i, j] = arr.mean() if reduce == "mean" else np.median(arr)
        return out

    def control_off_mask(self) -> np.ndarray:
        if self.control_off_col is None:
            return np.zeros(self.n_events, dtype=bool)
        return self.frame[self.control_off_col].to_numpy(dtype=bool)

    def subset(self, mask: Sequence[bool]) -> "EventTable":
        return EventTable(
            frame=self.frame.loc[np.asarray(mask)].reset_index(drop=True),
            knob_cols_recommended=self.knob_cols_recommended,
            knob_cols_used=self.knob_cols_used,
            ff_cols=self.ff_cols, y_cols=self.y_cols, y_time_cols=self.y_time_cols,
            time_col=self.time_col, control_off_col=self.control_off_col,
        )
