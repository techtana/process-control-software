"""Dense-but-shrunk ridge (REG-03).

Ridge is one of the shared cross-sectional solvers and lives in the sensitivity
core; it is re-exported here so the regression machine sees ridge, PLS and
elastic net side by side.
"""

from __future__ import annotations

from ..sensitivity.core import ridge_fit

__all__ = ["ridge_fit"]
