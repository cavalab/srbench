import pytest
import numpy as np

from bingo.evaluation.fitness_function import VectorBasedFunction as pyVectorBasedFunction
from bingo.evaluation.gradient_mixin import GradientMixin as pyGradientMixin, \
    VectorGradientMixin as pyVectorGradientMixin
from bingo.symbolic_regression.agraph.agraph import AGraph as pyAGraph

try:
    from bingocpp import GradientMixin as cppGradientMixin, \
        VectorGradientMixin as cppVectorGradientMixin, \
        VectorBasedFunction as cppVectorBasedFunction, \
        AGraph as cppAGraph
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
def gradient_mixin(engine):
    if engine == "Python":
        return pyGradientMixin
    return cppGradientMixin


@pytest.fixture
def vector_gradient_mixin(engine):
    if engine == "Python":
        return pyVectorGradientMixin
    return cppVectorGradientMixin


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
def dummy_individual(agraph):
    return agraph()


def test_gradient_mixin_cant_be_instanced(gradient_mixin):
    with pytest.raises(TypeError):
        _ = gradient_mixin()


def test_vector_gradient_mixin_cant_be_instanced():
    with pytest.raises(TypeError):
        _ = pyVectorGradientMixin()


def test_vector_gradient_mixin_cant_be_instanced_without_base_class():
    class MixinWithoutVectorBasedFunction(pyVectorGradientMixin):
        def get_fitness_vector_and_jacobian(self, individual):
            pass

    with pytest.raises(TypeError):
        _ = MixinWithoutVectorBasedFunction()


def test_vector_gradient_mixin_invalid_metric(vector_gradient_mixin):
    class NewParent:
        def __init__(self, training_data, metric):
            pass

    class VectorGradientMixinWithNewParent(vector_gradient_mixin, NewParent):
        def get_fitness_vector_and_jacobian(self, individual):
            pass

    with pytest.raises(ValueError):
        _ = VectorGradientMixinWithNewParent(training_data=None, metric="invalid metric")


def test_gradient_mixin_get_gradient_raises_not_implemented_error(mocker):
    mocker.patch.object(pyGradientMixin, "__abstractmethods__", new_callable=set)

    gradient_mixin = pyGradientMixin()
    with pytest.raises(NotImplementedError):
        gradient_mixin.get_fitness_and_gradient(None)


def test_vector_gradient_mixin_get_jacobian_raises_not_implemented_error(mocker):
    class VectorFitnessFunction(pyVectorGradientMixin, pyVectorBasedFunction):
        pass

    mocker.patch.object(VectorFitnessFunction, "__abstractmethods__", new_callable=set)

    vector_fitness_function = VectorFitnessFunction()
    with pytest.raises(NotImplementedError):
        vector_fitness_function.get_fitness_vector_and_jacobian(None)


@pytest.fixture
def vector_gradient_fitness_function(vector_gradient_mixin, vector_based_function):
    class VectorGradFitnessFunction(vector_gradient_mixin, vector_based_function):
        def __init__(self, metric):
            vector_gradient_mixin.__init__(self, metric=metric)
            vector_based_function.__init__(self, metric=metric)

        def evaluate_fitness_vector(self, individual):
            return np.array([-2, 0, 2])

        def get_fitness_vector_and_jacobian(self, individual):
            return self.evaluate_fitness_vector(individual), np.array([[0.5, 1, -0.5], [1, 2, 3]]).transpose()
    return VectorGradFitnessFunction


@pytest.mark.parametrize("metric, expected_fitness, expected_fit_grad",
                         [("mae", 4/3, [-1/3, 2/3]), ("mean absolute error", 4/3, [-1/3, 2/3]),
                          ("mse", 8/3, [-4/3, 8/3]), ("mean squared error", 8/3, [-4/3, 8/3]),
                          ("rmse", np.sqrt(8/3), [np.sqrt(3/8) * -2/3, np.sqrt(3/8) * 4/3]),
                          ("root mean squared error", np.sqrt(8/3), [np.sqrt(3/8) * -2/3, np.sqrt(3/8) * 4/3])])
def test_vector_gradient(vector_gradient_fitness_function, dummy_individual,
                         metric, expected_fitness, expected_fit_grad):
    vector_function = vector_gradient_fitness_function(metric)
    fitness, gradient = vector_function.get_fitness_and_gradient(dummy_individual)
    assert fitness == expected_fitness
    np.testing.assert_array_equal(gradient, expected_fit_grad)
