"""Component 5 — diagnostics and visualization (§9)."""

from .precedence import DiagnosticsEngine, Diagnosis, excitation_sufficiency, UNIDENTIFIABLE
from .ff_leakage import ff_leakage
from .gain_mismatch import gain_mismatch, gain_drift_monitor
from .qr_attribution import qr_attribution
from .variance_decomp import variance_decomposition
from .achievability import achievability_verdict

__all__ = ["DiagnosticsEngine", "Diagnosis", "excitation_sufficiency", "UNIDENTIFIABLE",
           "ff_leakage", "gain_mismatch", "gain_drift_monitor", "qr_attribution",
           "variance_decomposition", "achievability_verdict"]
