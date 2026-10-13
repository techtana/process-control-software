"""Explicit defaults for the §13 open decisions (never hidden)."""

from ..schema import RunConfig


def default_run_config(**overrides) -> RunConfig:
    """A RunConfig with every §13 decision set explicitly; unknown keys are rejected."""
    cfg = RunConfig()
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise AttributeError(f"unknown RunConfig field {k!r}")
        setattr(cfg, k, v)
    return cfg


__all__ = ["default_run_config", "RunConfig"]
