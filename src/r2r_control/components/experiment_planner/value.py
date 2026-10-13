"""EP-05 — proposal value: priority metric and estimated model improvement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ExperimentProposal:
    setpoint: Dict[str, float]            # adjustable inputs only (EP-02)
    excites_directions: List[str]
    information_gain: float               # already discounted for loop status (A6)
    predicted_model_improvement: float
    priority: float
    feasible: bool = True
    cost: float = 1.0

    def to_dict(self):
        return dict(self.__dict__)


def priority(information_gain: float, improvement: float) -> float:
    """Rank experiments so the marginal value of the next can be weighed against cost."""
    return information_gain * (1.0 + improvement)
