import numpy as np
import pytest

from nest_st import NEST, NESTConfig
from nest_st.nest import _resolve_dropout_config


def test_modes_are_explicit():
    with pytest.raises(ValueError, match="mode"):
        NESTConfig(mode="unknown")


def test_constructor_rejects_two_configuration_styles():
    config = NESTConfig(mode="dropout")
    with pytest.raises(TypeError, match="either config"):
        NEST(config, mode="dropout")


def test_positive_parameters_are_validated():
    with pytest.raises(ValueError, match="ae_epochs"):
        NESTConfig(mode="dropout", ae_epochs=0)


def test_dropout_defaults_do_not_depend_on_zero_rate():
    config = NESTConfig(mode="dropout", device="cpu")
    dense = np.ones((500, 100), dtype=np.float32)
    sparse = dense.copy()
    sparse[:, :95] = 0
    first = _resolve_dropout_config(config, dense)
    second = _resolve_dropout_config(config, sparse)
    assert first.n_neighbors == second.n_neighbors == 10
    assert first.latent_dim == second.latent_dim == 25
    assert first.n_pcs == second.n_pcs == 40
    assert first.learning_rate == second.learning_rate == 1e-3
    assert first.ae_epochs == second.ae_epochs == 1000
    assert first.gae_epochs == second.gae_epochs == 2000

