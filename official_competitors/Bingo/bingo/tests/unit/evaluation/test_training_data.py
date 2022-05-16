# Ignoring some linting rules in tests
# pylint: disable=missing-docstring
# pylint: disable=abstract-class-instantiated

import pytest
from bingo.evaluation.training_data import TrainingData


def test_training_data_cant_be_instanced():
    with pytest.raises(TypeError):
        _ = TrainingData()
