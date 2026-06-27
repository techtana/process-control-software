"""Cross-cutting foundations shared by every component (§4).

These are single implementations called by all components (§IF-06): the noise
floor, the counterfactual, and the validation framework each mean exactly one
thing system-wide.
"""

from . import stats, excitation, noise_floor, counterfactual, validation, engine

__all__ = ["stats", "excitation", "noise_floor", "counterfactual",
           "validation", "engine"]
