import numpy as np
import pytest

from r2r_control import synthetic
from r2r_control.components.simulator import Simulator, PlantConfig, ControllerConfig


@pytest.fixture
def dataset():
    return synthetic.make_dataset(seed=42, doe_age_days=20)


@pytest.fixture
def confounded_dataset():
    # no accidental excitation => strongly confounded inline knobs
    return synthetic.make_dataset(seed=11, override_fraction=0.0,
                                  control_off_episodes=0, n_knob=5, n_out=2)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def make_sim(truth, model_gain_scale=1.0, with_ff=True, plant=None, **ctrl):
    """Simulator on the ground-truth plant with the given controller settings."""
    return Simulator(M=truth.M, targets=truth.targets, lsl=truth.targets - 3,
                     usl=truth.targets + 3, u0=truth.u0,
                     FF_gain=truth.FF_gain if with_ff else None,
                     plant=plant or PlantConfig(meas_noise_std=0.2),
                     controller=ControllerConfig(**ctrl),
                     model_gain_scale=model_gain_scale, seed=0)
