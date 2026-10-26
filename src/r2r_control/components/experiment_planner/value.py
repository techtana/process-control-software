"""EP-05 — proposal value: priority metric, cost and estimated model improvement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ExperimentProposal:
    setpoint: Dict[str, float]            # adjustable inputs only, coded units (EP-02)
    excites_directions: List[str]
    information_gain: float               # already discounted for loop status (A6)
    predicted_model_improvement: float
    priority: float
    feasible: bool = True
    cost: float = 1.0
    setpoint_engineering: Optional[Dict[str, float]] = None   # when knob_bounds are given
    gap_alignment: float = 0.0            # share of the move along the gap directions
    purpose: str = ""

    def to_dict(self):
        return dict(self.__dict__)


def priority(information_gain: float, improvement: float, cost: float = 1.0) -> float:
    """Value per unit cost, so the marginal value of the next run can be weighed (EP-05)."""
    return information_gain * (1.0 + improvement) / max(cost, 1e-12)


def run_cost(x_coded, names: List[str], center: float, base: float = 1.0,
             knob_cost: Optional[Dict[str, float]] = None) -> float:
    """``base + Σ knob_cost_k · |x_k − center|`` (coded excursion from the design center)."""
    c = float(base)
    if knob_cost:
        for nm, v in zip(names, x_coded):
            c += float(knob_cost.get(nm, 0.0)) * abs(float(v) - center)
    return c


__all__ = ["ExperimentProposal", "priority", "run_cost"]
