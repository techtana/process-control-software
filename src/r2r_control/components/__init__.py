"""The six functional components (§5-§10).

Each depends on core + estimation (+ controller/metrics) but never on another
component (§11); they communicate through provenance-tracked artifacts, and the
pipeline wires them together.
"""

from .fb_identification import FBIdentifier, FBModel
from .experiment_planner import ExperimentPlanner, PlannerConfig, PlannerResult, ExperimentProposal
from .simulator import Simulator, PlantConfig, ControllerConfig, SimulationResult
from .controller_optimizer import ControllerOptimizer, SearchDimension, OptimizationResult
from .diagnostics import DiagnosticsEngine, Diagnosis
from .regression_machine import RegressionMachine, RegressionResult

__all__ = [
    "FBIdentifier", "FBModel",
    "ExperimentPlanner", "PlannerConfig", "PlannerResult", "ExperimentProposal",
    "Simulator", "PlantConfig", "ControllerConfig", "SimulationResult",
    "ControllerOptimizer", "SearchDimension", "OptimizationResult",
    "DiagnosticsEngine", "Diagnosis",
    "RegressionMachine", "RegressionResult",
]
