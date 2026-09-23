"""Noise-floor estimation (§4.4, NF-01..05)."""

from .analytic import (marchenko_pastur_edge, NoiseFloorResult, analytic_floor,
                       minimum_variance_benchmark, harris_index)
from .empirical import empirical_floor
from .cleaning import eigenvalue_clip, ledoit_peche_shrinkage
from .select import estimate

__all__ = [
    "marchenko_pastur_edge", "NoiseFloorResult", "analytic_floor",
    "minimum_variance_benchmark", "harris_index", "empirical_floor",
    "eigenvalue_clip", "ledoit_peche_shrinkage", "estimate",
]
