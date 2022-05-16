# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import numpy as np
import pytest
import dill

from bingo.symbolic_regression.agraph.operator_definitions import *
from bingo.symbolic_regression.agraph.agraph import AGraph as pyagraph

try:
    from bingocpp import AGraph as cppagraph
except ImportError:
    cppagraph = None

CPP_PARAM = pytest.param("c++",
                         marks=pytest.mark.skipif(not cppagraph,
                                                  reason='BingoCpp import '
                                                         'failure'))


@pytest.fixture(params=["Python", CPP_PARAM])
def engine(request):
    return request.param


@pytest.fixture
def agraph_implementation(engine):
    if engine == "Python":
        return pyagraph
    return cppagraph


@pytest.fixture
def addition_agraph(agraph_implementation):
    sample = agraph_implementation()
    sample.command_array = np.array([[VARIABLE, 0, 0],
                                     [VARIABLE, 1, 1],
                                     [ADDITION, 1, 0]], dtype=int)
    return sample


@pytest.fixture
def addition_agraph_with_constants(agraph_implementation):
    sample = agraph_implementation()
    sample.command_array = np.array([[CONSTANT, -1, -1],
                                     [CONSTANT, -1, -1],
                                     [ADDITION, 1, 0]], dtype=int)
    return sample


@pytest.fixture
def sin_agraph(agraph_implementation):
    sample = agraph_implementation()
    sample.command_array = np.array([[VARIABLE, 0, 0],
                                     [SIN, 0, 0]], dtype=int)
    return sample


def test_agraph_copy_and_distance(addition_agraph):
    agraph_2 = addition_agraph.copy()
    agraph_2.mutable_command_array[0, 0] = CONSTANT
    agraph_2.mutable_command_array[1, 1:] = 0
    agraph_2.mutable_command_array[2, 0] = SUBTRACTION

    assert addition_agraph.distance(agraph_2) == 4
    assert agraph_2.distance(addition_agraph) == 4


def test_agraph_complexity(addition_agraph, sin_agraph):
    assert addition_agraph.get_complexity() == 3
    assert sin_agraph.get_complexity() == 2


def test_local_opt_interface_1(addition_agraph_with_constants):
    assert addition_agraph_with_constants.needs_local_optimization()
    params = (2, 3)
    addition_agraph_with_constants.set_local_optimization_params(params)
    assert addition_agraph_with_constants.constants == params
    assert not addition_agraph_with_constants.needs_local_optimization()


def test_local_opt_interface_2(addition_agraph_with_constants):
    n_p = addition_agraph_with_constants.get_number_local_optimization_params()
    assert n_p == 2


def test_engine_identification(engine, addition_agraph):
    assert addition_agraph.engine == engine


def test_setting_fitness_updates_fit_set_cpp(addition_agraph):
    assert not addition_agraph.fit_set
    addition_agraph.fitness = 0
    assert addition_agraph.fit_set


def test_mutable_access_to_command_array_unsets_fitness(addition_agraph):
    addition_agraph.fitness = 0
    assert addition_agraph.fit_set
    _ = addition_agraph.mutable_command_array
    assert not addition_agraph.fit_set


def test_setting_command_array_unsets_fitness(addition_agraph):
    addition_agraph.fitness = 0
    assert addition_agraph.fit_set
    addition_agraph.command_array = np.ones((1, 3))
    assert not addition_agraph.fit_set


@pytest.mark.parametrize("format_", ["latex", "console", "stack"])
@pytest.mark.parametrize("raw", [True, False])
def test_can_get_formatted_strings(format_, raw, addition_agraph):
    string = addition_agraph.get_formatted_string(format_, raw=raw)
    assert isinstance(string, str)


def test_default_string_is_console_string(addition_agraph):
    console_string = addition_agraph.get_formatted_string("console")
    assert str(addition_agraph) == console_string


def test_can_pickle(addition_agraph):
    _ = dill.loads(dill.dumps(addition_agraph))


def test_can_get_and_set_fitness(addition_agraph):
    addition_agraph.fitness = 0.5
    assert addition_agraph.fitness == 0.5


def test_can_get_and_set_fit_set(addition_agraph):
    addition_agraph.fit_set = True
    assert addition_agraph.fit_set is True
    addition_agraph.fit_set = False
    assert addition_agraph.fit_set is False


def test_can_get_and_set_genetic_age(addition_agraph):
    addition_agraph.genetic_age = 10
    assert addition_agraph.genetic_age == 10
