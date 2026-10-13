"""Contracts: data model, _is_measured (DM-03/E14), shape contract (DM-04), artifact."""

import numpy as np
import pytest

from r2r_control.contracts import (_is_measured, check_shape_contract, ShapeContractError,
                                   ModelArtifact, LOAD_BEARING_ASSUMPTIONS, ProvenanceLog)


def test_is_measured_handles_all_nan_array():
    assert _is_measured(np.array([np.nan, np.nan])) is False     # E14: pd.notna says True
    assert _is_measured(np.array([np.nan, 1.0])) is True
    assert _is_measured(3.0) is True
    assert _is_measured(np.nan) is False
    assert _is_measured(None) is False
    assert _is_measured(np.array([])) is False


def test_effective_count_uses_is_measured(dataset):
    t = dataset.inline
    raw_notna = int(t.frame["y0"].notna().sum())
    assert t.effective_measured_count("y0") < raw_notna      # all-NaN cells excluded


def test_shape_contract_fails_loudly():
    check_shape_contract(np.zeros((3, 4)), 3, 4)
    with pytest.raises(ShapeContractError):
        check_shape_contract(np.zeros((3, 4)), 4, 3)


def test_recommended_used_alignment_enforced(dataset):
    from r2r_control.contracts import EventTable
    t = dataset.inline
    with pytest.raises(ShapeContractError):
        EventTable(frame=t.frame, knob_cols_recommended=t.knob_cols_recommended[:-1],
                   knob_cols_used=t.knob_cols_used, y_cols=t.y_cols)


def test_model_artifact_records_load_bearing_assumptions():
    art = ModelArtifact(kind="fb", coef=np.ones((2, 2)), param_cov=np.eye(2),
                        sigma2=np.ones(2), identifiable={"x": 1},
                        provenance=ProvenanceLog(), assumptions_relied_on=["A1", "A3"])
    ledger = art.assumption_ledger()
    assert set(ledger) == {"A1", "A3"}
    assert set(LOAD_BEARING_ASSUMPTIONS) == {"A1", "A2", "A3", "A4", "A5", "A6"}
    assert "assumptions_relied_on" in art.to_dict()
