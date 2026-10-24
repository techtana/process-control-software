"""Coded design space, engineering-unit mapping and feasibility (EP-04/08, E11, KI-6).

The planner designs in coded units ``[level_low, level_high]``. ``knob_bounds``
maps each adjustable knob's coded range onto its engineering range, so proposals
can be reported in process units and are feasible by construction. Integer
knobs are rounded in engineering units when bounds are given (in coded units
otherwise); categorical knobs snap to their allowed coded levels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ...core.gp import round_to_feasible


@dataclass
class CodedSpace:
    names: List[str]
    level_low: float = -1.0
    level_high: float = 1.0
    knob_bounds: Optional[Dict[str, Tuple[float, float]]] = None
    integer_dims: Optional[List[int]] = None
    categorical_levels: Optional[Dict[int, List[float]]] = None
    px_weighting: str = "uniform"

    def __post_init__(self):
        if self.px_weighting not in ("uniform", "central"):
            raise ValueError(f"px_weighting must be 'uniform' or 'central', got "
                             f"{self.px_weighting!r}")

    @property
    def d(self) -> int:
        return len(self.names)

    @property
    def center(self) -> float:
        return 0.5 * (self.level_low + self.level_high)

    def _affine(self):
        lo = np.full(self.d, self.level_low, dtype=float)
        hi = np.full(self.d, self.level_high, dtype=float)
        if self.knob_bounds:
            for j, nm in enumerate(self.names):
                if nm in self.knob_bounds:
                    lo[j], hi[j] = self.knob_bounds[nm]
        scale = (hi - lo) / (self.level_high - self.level_low)
        return lo, scale

    def to_engineering(self, X: np.ndarray) -> np.ndarray:
        lo, scale = self._affine()
        return lo + (np.atleast_2d(X) - self.level_low) * scale

    def to_coded(self, X_eng: np.ndarray) -> np.ndarray:
        lo, scale = self._affine()
        return self.level_low + (np.atleast_2d(X_eng) - lo) / np.where(scale != 0, scale, 1.0)

    def snap(self, X: np.ndarray) -> np.ndarray:
        """Make coded points feasible: integer and categorical dimensions (E11)."""
        X = np.atleast_2d(np.asarray(X, dtype=float)).copy()
        if self.integer_dims:
            if self.knob_bounds:
                E = self.to_engineering(X)
                E[:, self.integer_dims] = np.round(E[:, self.integer_dims])
                X = self.to_coded(E)
            else:
                X = round_to_feasible(X, integer_dims=self.integer_dims)
        if self.categorical_levels:
            X = round_to_feasible(X, categories=self.categorical_levels)
        return np.clip(X, self.level_low, self.level_high)

    def sample(self, rng: np.random.Generator, n: int, snap: bool = True) -> np.ndarray:
        X = rng.uniform(self.level_low, self.level_high, size=(n, self.d))
        return self.snap(X) if snap else X

    def reference(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Points distributed as p(x), over which prediction variance is averaged (EP-08)."""
        if self.px_weighting == "uniform":
            return rng.uniform(self.level_low, self.level_high, size=(n, self.d))
        sd = (self.level_high - self.level_low) / 6.0
        return np.clip(rng.normal(self.center, sd, size=(n, self.d)),
                       self.level_low, self.level_high)

    def setpoints(self, x: np.ndarray) -> Tuple[Dict[str, float], Optional[Dict[str, float]]]:
        coded = {nm: float(v) for nm, v in zip(self.names, np.ravel(x))}
        if not self.knob_bounds:
            return coded, None
        eng = self.to_engineering(np.ravel(x))[0]
        return coded, {nm: float(v) for nm, v in zip(self.names, eng)}
