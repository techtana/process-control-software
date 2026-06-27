"""Unified event-indexed data model and schema contract (§4.1, DM-01..05).

The whole system speaks one table: an event-indexed record where each
processing event carries a timestamp, the *recommended* and *used* knob
vectors, the feedforward inputs, and — when available — the post-processing
measurement(s) with their own measurement timestamp.  Metrology is sparse and
asynchronous, cells may be array-valued, and the recommended-vs-used
distinction is first-class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


def _is_measured(cell) -> bool:
    """The single measurement-presence predicate (DM-03).

    ``pandas.notna`` returns ``True`` for an array that contains only NaNs,
    which silently corrupts every measured-sample count.  This predicate is the
    one place the system decides whether a (possibly array-valued) cell carries
    a real measurement, and must gate every measurement-presence decision.
    """
    if cell is None:
        return False
    if isinstance(cell, float) and np.isnan(cell):
        return False
    arr = np.asarray(cell, dtype=float) if not np.isscalar(cell) else np.asarray([cell], dtype=float)
    if arr.size == 0:
        return False
    return bool(np.any(np.isfinite(arr)))


def is_measured(series_or_cell):
    """Vectorized form of :func:`_is_measured` over a Series, or scalar form."""
    if isinstance(series_or_cell, pd.Series):
        return series_or_cell.apply(_is_measured)
    return _is_measured(series_or_cell)


class ShapeContractError(ValueError):
    """Raised when an ``M``/dynamics artifact violates the n_out x n_knob contract (DM-04)."""


def check_shape_contract(M: np.ndarray, n_out: int, n_knob: int, where: str = "M") -> np.ndarray:
    """Enforce and verify the FB-model shape contract (DM-04).

    ``M`` maps knob deltas to output deltas and MUST be ``n_out x n_knob``.
    Fails loudly on mismatch at every interface that touches ``M``.
    """
    M = np.asarray(M, dtype=float)
    if M.shape != (n_out, n_knob):
        raise ShapeContractError(
            f"{where} has shape {M.shape}, expected ({n_out}, {n_knob}) "
            f"[n_out x n_knob]"
        )
    return M


@dataclass
class EventTable:
    """The unified event-indexed table (DM-01).

    Parameters
    ----------
    frame:
        A pandas DataFrame, one row per processing event.
    knob_cols_recommended / knob_cols_used:
        Two columns per knob: what the existing controller asked for vs what was
        physically applied.  Both retained end to end (DM-05).
    ff_cols:
        Feedforward inputs: pre-processing measurements, tool sensors, process
        configuration (including discrete configs such as product count).
    y_cols:
        Post-processing measurement column(s); cells may be scalar or array.
    y_time_cols:
        Per-output measurement-timestamp columns; metrology is asynchronous and
        may arrive after several subsequent events processed (DM-02).
    time_col:
        The event timestamp column.
    """

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

    # -- dimensions -------------------------------------------------------
    @property
    def n_events(self) -> int:
        return len(self.frame)

    @property
    def n_knob(self) -> int:
        return len(self.knob_cols_used)

    @property
    def n_out(self) -> int:
        return len(self.y_cols)

    # -- matrices ---------------------------------------------------------
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

    # -- recommended vs used (DM-05, EX-04) -------------------------------
    def recommended_used_divergence(self) -> np.ndarray:
        """Per-event, per-knob divergence used as a first-class signal (DM-05)."""
        return self.used_knobs() - self.recommended_knobs()

    # -- measurement presence (DM-02/03) ---------------------------------
    def measured_mask(self, y_col: Optional[str] = None) -> np.ndarray:
        """Boolean presence mask for one output, gated by :func:`_is_measured`."""
        cols = [y_col] if y_col is not None else self.y_cols
        masks = [self.frame[c].apply(_is_measured).to_numpy() for c in cols]
        if len(masks) == 1:
            return masks[0]
        return np.column_stack(masks)

    def effective_measured_count(self, y_col: Optional[str] = None) -> int:
        """Effective measured-sample count after ``_is_measured`` filtering (DQ-05)."""
        return int(np.sum(self.measured_mask(y_col)))

    def y_matrix(self, reduce: str = "mean") -> np.ndarray:
        """Reduce (possibly array-valued) measurement cells to a numeric matrix.

        Unmeasured cells become NaN; array cells are reduced by ``reduce``
        (mean/median) over finite entries only.
        """
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
            ff_cols=self.ff_cols,
            y_cols=self.y_cols,
            y_time_cols=self.y_time_cols,
            time_col=self.time_col,
            control_off_col=self.control_off_col,
        )
