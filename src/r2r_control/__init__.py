"""Run-to-Run Process Control Software (r2r_control).

A toolchain for building, simulating, tuning and diagnosing run-to-run (R2R)
process controllers from imperfect industrial data, organized around one idea:
variance accounting against a noise floor.

Layers (§16): contracts -> core -> estimation -> {controller, metrics} ->
components -> pipeline. Components 1 and 6 share a sensitivity core and the
cross-cutting services but are NOT one estimator class (§1.3).
"""

from . import contracts, config, core, estimation, controller, metrics, components, pipeline
from . import synthetic

__all__ = ["contracts", "config", "core", "estimation", "controller", "metrics",
           "components", "pipeline", "synthetic"]
__version__ = "0.2.0"
