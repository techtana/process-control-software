"""Minimum effective-sample gate for real, sparse metrology (SIM-06, E10).

Cpk on a handful of points is very noisy and a Harris index from a short
closed-loop record is biased. Any metric whose effective measured-sample count
(after ``_is_measured``, §DM-03) falls below the configured floor is suppressed
and reported as "insufficient samples" rather than as a falsely precise number.
"""

INSUFFICIENT = "insufficient samples"


def min_sample_gate(n_effective: int, floor: int) -> bool:
    """True when the metric must be suppressed (under-powered)."""
    return n_effective < floor
