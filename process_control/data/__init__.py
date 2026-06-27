"""Data model, schema contract, and the data-quality gatekeeper (§4.1, §4.2)."""

from .schema import EventTable, check_shape_contract, _is_measured, is_measured
from .quality import DataQualityLayer, QualityReport

__all__ = ["EventTable", "check_shape_contract", "_is_measured", "is_measured",
           "DataQualityLayer", "QualityReport"]
