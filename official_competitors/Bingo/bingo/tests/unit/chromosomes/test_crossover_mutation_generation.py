# Ignoring some linting rules in tests
# pylint: disable=missing-docstring

import pytest
from bingo.chromosomes.crossover import Crossover
from bingo.chromosomes.mutation import Mutation
from bingo.chromosomes.generator import Generator


def test_base_crossover_instantiation_fails():
    with pytest.raises(TypeError):
        _ = Crossover()


def test_base_mutation_instantiation_fails():
    with pytest.raises(TypeError):
        _ = Mutation()


def test_base_generator_instantiation_fails():
    with pytest.raises(TypeError):
        _ = Generator()