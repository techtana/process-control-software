"""Provenance and auditability primitives (NFR-03, DQ-08).

Every model artifact in this system carries the record of *why* it is what it
is: which data was used, down-weighted, or rejected and on what evidence; which
noise floor was applied and why; the validation scheme and its error; and the
assumptions that were checked.  These primitives are the common currency for
that record so a future session or reviewer can reconstruct the reasoning
without re-establishing context.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


class Verdict(str, Enum):
    """The use / down-weight / reject decision of the data-quality layer (DQ-03/06)."""

    USE = "use"
    DOWN_WEIGHT = "down_weight"
    REJECT = "reject"


@dataclass
class Decision:
    """A single use/down-weight/reject decision with its triggering diagnostics.

    Persisted with the resulting artifact so any downstream consumer can audit
    why a datum was or was not used (DQ-08).
    """

    source: str
    verdict: Verdict
    rationale: str
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["verdict"] = self.verdict.value
        return d


@dataclass
class Assumption:
    """A named assumption, whether it held, and the evidence."""

    name: str
    holds: bool
    detail: str = ""
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProvenanceLog:
    """Append-only audit log attached to every artifact (NFR-03).

    Records data-usage decisions, checked assumptions, the noise floor used and
    why, the validation scheme and its error, and free-form notes.
    """

    decisions: List[Decision] = field(default_factory=list)
    assumptions: List[Assumption] = field(default_factory=list)
    noise_floor: Dict[str, Any] = field(default_factory=dict)
    validation: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    seed: Optional[int] = None

    def decide(
        self,
        source: str,
        verdict: Verdict,
        rationale: str,
        diagnostics: Optional[Dict[str, Any]] = None,
        weight: float = 1.0,
    ) -> Decision:
        dec = Decision(source, verdict, rationale, diagnostics or {}, weight)
        self.decisions.append(dec)
        return dec

    def assume(
        self,
        name: str,
        holds: bool,
        detail: str = "",
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> Assumption:
        a = Assumption(name, holds, detail, diagnostics or {})
        self.assumptions.append(a)
        return a

    def note(self, msg: str) -> None:
        self.notes.append(msg)

    def record_noise_floor(self, kind: str, value: Any, reason: str, **extra: Any) -> None:
        """NF-04: record which floor was used (analytic vs empirical) and why."""
        self.noise_floor = {"kind": kind, "value": value, "reason": reason, **extra}

    def record_validation(self, scheme: str, error: Any, **extra: Any) -> None:
        """NFR-03: record the validation scheme and its (honest, forward) error."""
        self.validation = {"scheme": scheme, "error": error, **extra}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decisions": [d.to_dict() for d in self.decisions],
            "assumptions": [a.to_dict() for a in self.assumptions],
            "noise_floor": self.noise_floor,
            "validation": self.validation,
            "notes": list(self.notes),
            "config": self.config,
            "seed": self.seed,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=_json_default)

    def merge(self, other: "ProvenanceLog", prefix: str = "") -> None:
        """Fold another log into this one (used when fusing data sources, DQ-07)."""
        for d in other.decisions:
            d2 = Decision(
                source=f"{prefix}{d.source}" if prefix else d.source,
                verdict=d.verdict,
                rationale=d.rationale,
                diagnostics=d.diagnostics,
                weight=d.weight,
            )
            self.decisions.append(d2)
        self.assumptions.extend(other.assumptions)
        self.notes.extend(f"{prefix}{n}" if prefix else n for n in other.notes)


def _json_default(obj: Any) -> Any:
    """Make numpy scalars/arrays JSON-serializable for audit dumps."""
    try:
        import numpy as np

        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
    except Exception:  # pragma: no cover - numpy always present in this stack
        pass
    if isinstance(obj, Verdict):
        return obj.value
    if isinstance(obj, Enum):
        return obj.value
    return str(obj)
