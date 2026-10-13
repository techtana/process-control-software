"""Component-6-only high-dimensional regularized-selection layer (§1.3, REG-03/06)."""

from .pls import pls_fit
from .elastic_net import elastic_net_fit
from .ridge import ridge_fit
from .stability_selection import stability_selection, StabilityResult, infer_groups

__all__ = ["pls_fit", "elastic_net_fit", "ridge_fit", "stability_selection",
           "StabilityResult", "infer_groups"]
