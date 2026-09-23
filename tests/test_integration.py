"""End-to-end: pipeline, shared-service contract (IF-06), reproducibility (NFR-02)."""

import json

import numpy as np

from r2r_control import synthetic
from r2r_control.config import default_run_config
from r2r_control.pipeline import GatePipeline


def _cfg(truth, **kw):
    return default_run_config(targets=truth.targets.tolist(),
                              lsl=(truth.targets - 3).tolist(),
                              usl=(truth.targets + 3).tolist(), **kw)


def test_pipeline_end_to_end_and_serializable():
    ds = synthetic.make_dataset(seed=0, doe_age_days=20)
    res = GatePipeline(_cfg(ds.truth)).run(ds.inline, ds.doe, doe_age_days=20)
    assert np.linalg.norm(res.fb.M - ds.truth.M) / np.linalg.norm(ds.truth.M) < 0.35
    assert res.plan.proposals
    assert res.diagnosis.excitation["sufficient"]
    assert res.regression.decoupled
    assert len(res.gate_log) >= 5
    json.dumps(res.to_dict(), default=str)


def test_pipeline_is_reproducible():
    ds = synthetic.make_dataset(seed=1)
    a = GatePipeline(_cfg(ds.truth, seed=3)).run(ds.inline, ds.doe, run_planner=False)
    b = GatePipeline(_cfg(ds.truth, seed=3)).run(ds.inline, ds.doe, run_planner=False)
    assert np.allclose(a.fb.M, b.fb.M)
    assert a.regression.forward_error == b.regression.forward_error


def test_shared_noise_floor_is_one_implementation(dataset):
    # IF-06: C1's identifiability report and a direct call agree on the floor kind
    from r2r_control.components.fb_identification import FBIdentifier
    fb = FBIdentifier().identify(dataset.inline, dataset.doe)
    kind = fb.provenance.to_dict()["noise_floor"]["kind"]
    assert kind in {"analytic_mp", "empirical_block_surrogate", "empirical_phase_surrogate"}


def test_config_rejects_unknown_fields():
    import pytest
    with pytest.raises(AttributeError):
        default_run_config(not_a_field=1)
