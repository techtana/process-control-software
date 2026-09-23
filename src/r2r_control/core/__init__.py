"""Cross-cutting shared services — one implementation each (§4, §IF-06).

The noise floor, the counterfactual, and the validation framework each mean
exactly one thing system-wide because they are single implementations called by
all components.
"""

from . import (stats, gp, excitation, noise_floor, counterfactual, validation,
               data_quality)

__all__ = ["stats", "gp", "excitation", "noise_floor", "counterfactual",
           "validation", "data_quality"]
