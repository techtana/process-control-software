"""Estimation layer encoding the corrected unification (§1.3).

`sensitivity/` is the shared cross-sectional core (Component 1's gain and
Component 6); `dynamics/` is Component-1-only; `regularized/` is
Component-6-only. They are NOT one estimator class — they share the core and the
cross-cutting services.
"""

from . import sensitivity, dynamics, regularized

__all__ = ["sensitivity", "dynamics", "regularized"]
