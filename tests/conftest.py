import numpy as np
import pytest

from process_control import synthetic


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
