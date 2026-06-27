"""The six functional components (§5-§10).

Components 1 and 6 are two configurations of one estimation engine (§REG-01);
Components 3, 4, 5 form the simulate -> optimize -> diagnose loop; Component 2
turns identifiability gaps into prioritized, justified experiment proposals.
"""

from .component1_identification import FBIdentifier, FBModel
from .component2_planner import ExperimentPlanner, PlannerConfig, PlannerResult
from .component3_simulator import (Simulator, PlantConfig, ControllerConfig,
                                  Metrics, SimulationResult, compute_metrics)
from .component4_optimizer import ControllerOptimizer, SearchDimension, OptimizationResult
from .component5_diagnostics import DiagnosticsEngine, Diagnosis
from .component6_regression import RegressionMachine, RegressionResult

__all__ = [
    "FBIdentifier", "FBModel",
    "ExperimentPlanner", "PlannerConfig", "PlannerResult",
    "Simulator", "PlantConfig", "ControllerConfig", "Metrics",
    "SimulationResult", "compute_metrics",
    "ControllerOptimizer", "SearchDimension", "OptimizationResult",
    "DiagnosticsEngine", "Diagnosis",
    "RegressionMachine", "RegressionResult",
]
