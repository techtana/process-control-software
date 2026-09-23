"""Declarative configuration (NFR-02, §13)."""

from .schema import RunConfig
from .defaults import default_run_config

__all__ = ["RunConfig", "default_run_config"]
