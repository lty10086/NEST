import pytest

from nest_st import NEST, NESTConfig


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

