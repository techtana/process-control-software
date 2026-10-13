"""Declarative, seeded run configuration (NFR-02, §13).

The open design decisions of §13 are explicit configuration, never silent
defaults: each field is a decision a human owns, and the default only makes the
current choice visible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RunConfig:
    seed: int = 0                                   # NFR-02 reproducibility

    # §13.2 counterfactual baseline u0 (CF-02, A2)
    allow_time_varying_u0: bool = False

    # §13.3 spec inputs (SIM-06) + finite-sample gate (E10)
    targets: Optional[List[float]] = None
    lsl: Optional[List[float]] = None
    usl: Optional[List[float]] = None
    min_samples: int = 8

    # §13.4 M uncertainty for MC bands / phantom-FF guard (CF-03, SIM-02, A3)
    rel_uncertainty_M: float = 0.1

    # §13.6 experiment-planner choices (EP-08, A6, E11)
    planner_objective: str = "D"                    # 'D' | 'I'
    loop_status: str = "open"                       # 'open' | 'closed'
    budget: int = 8
    integer_dims: Optional[List[int]] = None

    # §13.7 regression target (REG-04)
    regression_target: str = "raw"                  # 'raw' | 'innovation'

    # FB estimator
    estimator_fb: str = "svd"                       # 'svd' | 'ridge'
    notes: List[str] = field(default_factory=list)

    def to_dict(self):
        return dict(self.__dict__)
