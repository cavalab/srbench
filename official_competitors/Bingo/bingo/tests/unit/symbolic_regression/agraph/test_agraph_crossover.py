# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import numpy as np
import pytest

from bingo.symbolic_regression.agraph.agraph import AGraph
from bingo.symbolic_regression.agraph.crossover import AGraphCrossover


@pytest.fixture
def sample_agraph_zeros(mocker):
    stack = np.zeros((10, 3), dtype=int)
    sample = mocker.create_autospec(AGraph)
    type(sample).command_array = mocker.PropertyMock(return_value=stack.copy())
    type(sample).mutable_command_array = \
        mocker.PropertyMock(return_value=stack.copy())
    type(sample).genetic_age = mocker.PropertyMock(return_value=0)
    sample.copy.return_value = sample
    return sample


@pytest.fixture
def sample_agraph_ones(mocker):
    stack = np.ones((10, 3), dtype=int)
    sample = mocker.create_autospec(AGraph)
    type(sample).command_array = mocker.PropertyMock(return_value=stack.copy())
    type(sample).mutable_command_array = \
        mocker.PropertyMock(return_value=stack.copy())
    type(sample).genetic_age = mocker.PropertyMock(return_value=1)
    sample.copy.return_value = sample
    return sample

@pytest.fixture(params=[('sample_agraph_zeros', 'sample_agraph_ones'),
                        ('sample_agraph_ones', 'sample_agraph_zeros')])
def crossover_parents(request):
    return (request.getfixturevalue(request.param[0]),
            request.getfixturevalue(request.param[1]))


def test_crossover_is_single_point(crossover_parents):
    crossover = AGraphCrossover()
    child_1, child_2 = crossover(crossover_parents[0], crossover_parents[1])
    stack_p1 = crossover_parents[0].command_array
    stack_p2 = crossover_parents[1].command_array
    stack_c1 = child_1.mutable_command_array
    stack_c2 = child_2.mutable_command_array
    assert np.logical_xor(stack_c1 == stack_p1, stack_c1 == stack_p2).all()
    assert np.logical_xor(stack_c2 == stack_p1, stack_c2 == stack_p2).all()


def test_crossover_genetic_age(mocker):
    crossover = AGraphCrossover()
    parent_1 = mocker.MagicMock(genetic_age=1)
    type(parent_1.command_array).shape = \
        mocker.PropertyMock(return_value=[3, 3])
    parent_1.copy.return_value = mocker.MagicMock(genetic_age=1)
    parent_2 = mocker.MagicMock(genetic_age=2)
    parent_1.copy.return_value = mocker.MagicMock(genetic_age=2)
    child_1, child_2 = crossover(parent_1, parent_2)

    assert child_1.genetic_age == 2
    assert child_2.genetic_age == 2

