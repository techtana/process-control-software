"""Run-to-Run Process Control Software.

A cohesive toolchain for building, simulating, tuning, and diagnosing
run-to-run (R2R) process controllers from imperfect industrial data, organized
around one unifying idea: variance accounting against a noise floor.

The six functional components share a single conceptual core — the data model,
the data-quality layer, the excitation/confounding analysis, the noise-floor
estimator, the counterfactual-reconstruction service, and the validation
framework — rather than re-deriving them independently (§1, §IF-06).
"""

from . import provenance, synthetic
from . import data, foundations, components

__all__ = ["provenance", "synthetic", "data", "foundations", "components"]
__version__ = "0.1.0"
