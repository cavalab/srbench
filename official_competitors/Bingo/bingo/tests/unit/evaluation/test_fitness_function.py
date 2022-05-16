# Ignoring some linting rules in tests
# pylint: disable=missing-docstring

import pytest
import numpy as np
from bingo.evaluation.fitness_function import FitnessFunction as pyFitnessFunction,\
    VectorBasedFunction as pyVectorBasedFunction
from bingo.symbolic_regression.agraph.agraph import AGraph as pyAGraph
from bingo.evaluation.training_data import TrainingData as pyTrainingData

try:
    from bingocpp import AGraph as cppAGraph, \
        FitnessFunction as cppFitnessFunction, \
        VectorBasedFunction as cppVectorBasedFunction, \
        TrainingData as cppTrainingData
    bingocpp = True
except ImportError:
    bingocpp = False

CPP_PARAM = pytest.param("Cpp",
                         marks=pytest.mark.skipif(not bingocpp,
                                                  reason='BingoCpp import '
                                                         'failure'))


@pytest.fixture(params=["Python", CPP_PARAM])
def engine(request):
    return request.param


@pytest.fixture
def fitness_function(engine):
    if engine == "Python":
        return pyFitnessFunction
    return cppFitnessFunction


@pytest.fixture
def vector_based_function(engine):
    if engine == "Python":
        return pyVectorBasedFunction
    return cppVectorBasedFunction


@pytest.fixture
def agraph(engine):
    if engine == "Python":
        return pyAGraph
    return cppAGraph


@pytest.fixture
def training_data(engine):
    if engine == "Python":
        return pyTrainingData
    return cppTrainingData


@pytest.fixture
def dummy_individual(agraph):
    return agraph()


def test_fitness_function_cant_be_instanced():
    with pytest.raises(TypeError):
        _ = pyFitnessFunction()


def test_fitness_function_has_eval_count_and_data(engine, mocker, fitness_function, training_data):
    if engine == "Python":
        mocker.patch.object(fitness_function, "__abstractmethods__",
                            new_callable=set)
        mocker.patch.object(training_data, "__abstractmethods__",
                            new_callable=set)
    training_data = training_data()
    fit_func = fitness_function(training_data)

    assert fit_func.eval_count == 0
    assert fit_func.training_data is training_data


@pytest.mark.parametrize("metric, expected_fit", [("mae", 1.2),
                                                  ("mean absolute error", 1.2),
                                                  ("mse", 2.0),
                                                  ("mean squared error", 2.0),
                                                  ("rmse", np.sqrt(2.0)),
                                                  ("root mean squared error", np.sqrt(2.0))])
def test_vector_based_function_metrics(engine, mocker, vector_based_function, metric, expected_fit, dummy_individual):
    if engine == "Python":
        mocker.patch.object(vector_based_function, "__abstractmethods__",
                            new_callable=set)
    mocker.patch.object(vector_based_function, "evaluate_fitness_vector",
                        return_value=[-2, -1, 0, 1, 2])
    fit_func = vector_based_function(metric=metric)
    
    assert fit_func(dummy_individual) == pytest.approx(expected_fit)
    fit_func.evaluate_fitness_vector.assert_called_once_with(dummy_individual)


def test_vector_based_function_invalid_metric(engine, mocker, vector_based_function):
    if engine == "Python":
        mocker.patch.object(vector_based_function, "__abstractmethods__",
                            new_callable=set)
    mocker.patch.object(vector_based_function, "evaluate_fitness_vector",
                        return_value=[-2, -1, 0, 1, 2])
    with pytest.raises(ValueError):
        _ = vector_based_function(metric="invalid metric")


@pytest.mark.parametrize("metric", ["mae", "mse", "rmse",
                                    "mean absolute error",
                                    "mean squared error",
                                    "root mean squared error"])
def test_vector_based_function_with_nan(engine, mocker, vector_based_function, agraph, metric, dummy_individual):
    if engine == "Python":
        mocker.patch.object(vector_based_function, "__abstractmethods__",
                            new_callable=set)
    mocker.patch.object(vector_based_function, "evaluate_fitness_vector",
                        return_value=[np.nan, -1, 0, 1, 2])
    fit_func = vector_based_function(metric=metric)

    assert np.isnan(fit_func(dummy_individual))
