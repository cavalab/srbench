# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import numpy as np
import pytest
import dill

from copy import copy

from bingo.symbolic_regression.explicit_regression \
    import ExplicitTrainingData as pyExplicitTrainingData, \
           ExplicitRegression as pyExplicitRegression
from bingo.symbolic_regression.equation import Equation as pyEquation
try:
    from bingocpp import ExplicitTrainingData as cppExplicitTrainingData, \
                         ExplicitRegression as cppExplicitRegression, \
                         Equation as cppEquation
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
def explicit_training_data(engine):
    if engine == "Python":
        return pyExplicitTrainingData
    return cppExplicitTrainingData


@pytest.fixture
def explicit_regression(engine):
    if engine == "Python":
        return pyExplicitRegression
    return cppExplicitRegression


@pytest.fixture
def equation(engine):
    if engine == "Python":
        return pyEquation
    return cppEquation


@pytest.fixture
def sample_training_data(explicit_training_data):
    x = np.arange(10, dtype=float)
    y = np.arange(1, 11, dtype=float)
    return explicit_training_data(x, y)


@pytest.fixture
def sample_regression(sample_training_data, explicit_regression):
    return explicit_regression(sample_training_data, relative=False)


@pytest.fixture
def sample_regression_relative(sample_training_data, explicit_regression):
    return explicit_regression(sample_training_data, relative=True)


@pytest.fixture
def sample_equation(equation):
    class SampleEqu(equation):
        def evaluate_equation_at(self, x):
            return np.ones((10, 1), dtype=float)

        def evaluate_equation_with_x_gradient_at(self, x):
            pass

        def evaluate_equation_with_local_opt_gradient_at(self, x):
            pass

        def get_complexity(self):
            pass

        def get_latex_string(self):
            pass

        def get_console_string(self):
            pass

        def __str__(self):
            pass

        def distance(self, _chromosome):
            return 0

    return SampleEqu()


@pytest.fixture
def sample_constants():
    return [2.0, 3.0]


@pytest.fixture
def sample_differentiable_equation(equation, sample_constants):
    class SampleDiffEqu(equation):
        def __init__(self):
            super().__init__()
            self.constants = sample_constants

        def evaluate_equation_at(self, x):
            return self.constants[0] * x + self.constants[1]

        def evaluate_equation_with_x_gradient_at(self, x):
            return self.constants[0]

        def evaluate_equation_with_local_opt_gradient_at(self, x):
            return self.evaluate_equation_at(x), \
                   np.concatenate((x, np.ones(len(x)).reshape(-1, 1)), axis=1)

        def get_complexity(self):
            pass

        def get_latex_string(self):
            pass

        def get_console_string(self):
            pass

        def __str__(self):
            pass

        def distance(self, _chromosome):
            return 0

    return SampleDiffEqu()


@pytest.fixture
def expected_differentiable_equation_output(sample_constants):
    return sample_constants[0] * np.arange(10, dtype=float) \
           + sample_constants[1]


@pytest.mark.parametrize("three_dim", ["x", "y"])
def test_raises_error_on_training_data_with_high_dims(explicit_training_data,
                                                      three_dim):
    x = np.zeros(10)
    y = np.zeros(10)
    if three_dim == "x":
        x = x.reshape((-1, 1, 1))
    else:
        y = y.reshape((-1, 1, 1))

    with pytest.raises(TypeError):
        explicit_training_data(x, y)


def test_training_data_xy(explicit_training_data):
    x = np.zeros(10)
    y = np.ones(10)
    etd = explicit_training_data(x, y)
    np.testing.assert_array_equal(etd.x, np.zeros((10, 1)))
    np.testing.assert_array_equal(etd.y, np.ones((10, 1)))


def test_training_data_xy_read_only(explicit_training_data):
    x = np.zeros(10)
    y = np.ones(10)
    etd = explicit_training_data(x, y)
    _ = etd.x
    _ = etd.y

    with pytest.raises(AttributeError):
        etd.x = 1

    with pytest.raises(AttributeError):
        etd.y = 1


def test_training_data_slicing(sample_training_data):
    indices = [2, 4, 6, 8]
    sliced_etd = sample_training_data[indices]
    expected_x = np.array(indices).reshape((-1, 1))
    expected_y = np.array(indices).reshape((-1, 1)) + 1
    np.testing.assert_array_equal(sliced_etd.x, expected_x)
    np.testing.assert_array_equal(sliced_etd.y, expected_y)


@pytest.mark.parametrize("num_elements", range(1, 4))
def test_training_data_len(explicit_training_data, num_elements):
    x = np.arange(num_elements)
    y = np.arange(num_elements)
    etd = explicit_training_data(x, y)
    assert len(etd) == num_elements


def test_explicit_regression(sample_regression, sample_equation):
    assert sample_regression.eval_count == 0
    fit_vec = sample_regression.evaluate_fitness_vector(sample_equation)
    expected_fit_vec = 1 - np.arange(1, 11)
    np.testing.assert_array_equal(fit_vec, expected_fit_vec)
    assert sample_regression.eval_count == 1


def test_explicit_regression_relative(sample_regression_relative,
                                      sample_equation):
    assert sample_regression_relative.eval_count == 0
    fit_vec = sample_regression_relative.evaluate_fitness_vector(
        sample_equation)
    expected_fit_vec = (1 - np.arange(1, 11)) / np.arange(1, 11)
    np.testing.assert_array_equal(fit_vec, expected_fit_vec)
    assert sample_regression_relative.eval_count == 1


@pytest.mark.parametrize("metric, expected_fit, expected_grad",
                         [("mae", 6.5, np.array([4.5, 1])),
                          ("mse", 50.5, 2 * np.array([37.5, 6.5])),
                          ("rmse", np.sqrt(50.5),
                           1/np.sqrt(50.5) * np.array([37.5, 6.5]))])
def test_explicit_regression_get_fit_and_grad(
        explicit_regression, sample_training_data,
        sample_differentiable_equation, metric, expected_fit, expected_grad):
    sample_regression = explicit_regression(sample_training_data, metric,
                                            relative=False)
    assert sample_regression.eval_count == 0
    fitness, gradient = sample_regression.get_fitness_and_gradient(
        sample_differentiable_equation)

    assert fitness == expected_fit
    np.testing.assert_array_equal(gradient, expected_grad)
    assert sample_regression.eval_count == 1


@pytest.mark.parametrize("metric, expected_fit, expected_grad",
                         [("mae", 1.293, np.array([0.707, 0.293])),
                          ("mse", 1.741, 2 * np.array([0.845, 0.448])),
                          ("rmse", 1.319, 0.758 * np.array([0.845, 0.448]))])
def test_explicit_regression_relative_get_fit_and_grad(
        explicit_regression, sample_training_data,
        sample_differentiable_equation, metric, expected_fit, expected_grad):
    sample_regression_relative = explicit_regression(sample_training_data,
                                                     metric, relative=True)
    assert sample_regression_relative.eval_count == 0
    fitness, gradient = sample_regression_relative.get_fitness_and_gradient(
            sample_differentiable_equation)

    assert fitness == pytest.approx(expected_fit, 0.001)
    np.testing.assert_array_almost_equal(gradient, expected_grad, 3)
    assert sample_regression_relative.eval_count == 1


def test_differentiable_explicit_regression(
        sample_regression, sample_differentiable_equation,
        expected_differentiable_equation_output):
    assert sample_regression.eval_count == 0
    fit_vec = sample_regression.evaluate_fitness_vector(
            sample_differentiable_equation)
    expected_fit_vec = expected_differentiable_equation_output \
        - np.arange(1, 11)
    np.testing.assert_array_equal(fit_vec, expected_fit_vec)
    assert sample_regression.eval_count == 1


def test_differentiable_explicit_regression_relative(
        sample_regression_relative, sample_differentiable_equation,
        expected_differentiable_equation_output):
    assert sample_regression_relative.eval_count == 0
    fit_vec = sample_regression_relative.evaluate_fitness_vector(
            sample_differentiable_equation)
    expected_fit_vec = (expected_differentiable_equation_output
                        - np.arange(1, 11)) / np.arange(1, 11)
    np.testing.assert_array_equal(fit_vec, expected_fit_vec)
    assert sample_regression_relative.eval_count == 1


def test_explicit_regression_get_fit_vec_and_jac(
        sample_regression, sample_differentiable_equation,
        expected_differentiable_equation_output):
    assert sample_regression.eval_count == 0
    fit_vec, fit_jac = sample_regression.get_fitness_vector_and_jacobian(
            sample_differentiable_equation)

    expected_fit_vec = expected_differentiable_equation_output \
                       - np.arange(1, 11)
    expected_jac = np.vstack((np.arange(10, dtype=float), np.ones(10))).T

    np.testing.assert_array_equal(fit_vec, expected_fit_vec)
    np.testing.assert_array_equal(fit_jac, expected_jac)
    assert sample_regression.eval_count == 1


def test_explicit_regression_relative_get_fit_vec_and_jac(
        sample_regression_relative, sample_differentiable_equation,
        expected_differentiable_equation_output):
    assert sample_regression_relative.eval_count == 0
    fit_vec, fit_jac = \
        sample_regression_relative.get_fitness_vector_and_jacobian(
                sample_differentiable_equation)

    expected_fit_vec = (expected_differentiable_equation_output
                        - np.arange(1, 11)) / np.arange(1, 11)
    expected_jac = np.vstack((np.arange(10, dtype=float) / np.arange(1, 11),
                              np.ones(10) / np.arange(1, 11))).transpose()

    np.testing.assert_array_equal(fit_vec, expected_fit_vec)
    np.testing.assert_array_equal(fit_jac, expected_jac)
    assert sample_regression_relative.eval_count == 1


def test_can_pickle(sample_regression):
    _ = dill.loads(dill.dumps(sample_regression))


def test_can_copy(sample_regression):
    copied = copy(sample_regression)

    np.testing.assert_array_equal(copied.training_data.x, sample_regression.training_data.x)
    assert copied.eval_count == sample_regression.eval_count
