"""Shared cross-sectional sensitivity core (§1.3)."""

from .core import (SensitivityCore, FittedSensitivity, IdentifiableDirectionReport,
                   ridge_fit, svd_truncated_fit)

__all__ = ["SensitivityCore", "FittedSensitivity", "IdentifiableDirectionReport",
           "ridge_fit", "svd_truncated_fit"]
