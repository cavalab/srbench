# Ignoring some linting rules in tests
# pylint: disable=missing-docstring
# pylint: disable=abstract-class-instantiated
import pytest
from bingo.variation.variation import Variation


def test_variation_cant_be_instanced():
    with pytest.raises(TypeError):
        _ = Variation()
