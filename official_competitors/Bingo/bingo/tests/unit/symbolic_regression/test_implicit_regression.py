# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import numpy as np
import pytest
import dill

from copy import copy

from bingo.symbolic_regression.implicit_regression \
    import ImplicitTrainingData as pyImplicitTrainingData, \
           ImplicitRegression as pyImplicitRegression
from bingo.symbolic_regression.equation import Equation as pyEquation
try:
    from bingocpp import ImplicitTrainingData as cppImplicitTrainingData, \
                         ImplicitRegression as cppImplicitRegression, \
                         Equation as cppEquation
    bingocpp = True
except ImportError:
    bingocpp = False

CPP_PARAM = pytest.param("Cpp",
                         marks=pytest.mark.skipif(not bingocpp,
                                                  reason='BingoCpp import '
                                                         'failure'))


@pytest.fixture(params=["python", CPP_PARAM])
def engine(request):
    return request.param


@pytest.fixture
def implicit_training_data(engine):
    if engine == "python":
        return pyImplicitTrainingData
    return cppImplicitTrainingData


@pytest.fixture
def implicit_regression(engine):
    if engine == "python":
        return pyImplicitRegression
    return cppImplicitRegression


@pytest.fixture
def equation(engine):
    if engine == "python":
        return pyEquation
    return cppEquation


@pytest.fixture
def sample_training_data(implicit_training_data):
    x = np.arange(30, dtype=float).reshape((10, 3))
    dx_dt = np.array([[3, 2, 1]]*10, dtype=float)
    return implicit_training_data(x, dx_dt)


@pytest.fixture
def sample_implicit_regression(implicit_regression, sample_training_data):
    return implicit_regression(sample_training_data)


@pytest.fixture
def sample_equation(equation):

    class SampleEqu(equation):
        def evaluate_equation_at(self, x):
            pass

        def evaluate_equation_with_x_gradient_at(self, x):
            sample_df_dx = np.array([[1, 0, -1]] * 10)
            return np.ones((10, 3), dtype=float), sample_df_dx

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


@pytest.mark.parametrize("three_dim", ["x", "dx_dt"])
def test_raises_error_on_training_data_with_high_dims(implicit_training_data,
                                                      three_dim):
    x = np.zeros(10)
    dx_dt = np.zeros((10, 1))
    if three_dim == "x":
        x = x.reshape((-1, 1, 1))
    else:
        dx_dt = dx_dt.reshape((-1, 1, 1))

    with pytest.raises(TypeError):
        implicit_training_data(x, dx_dt)


def test_training_data_x_dxdt(implicit_training_data):
    x = np.zeros(10)
    dx_dt = np.ones((10, 1))
    itd = implicit_training_data(x, dx_dt)
    np.testing.assert_array_equal(itd.x, np.zeros((10, 1)))
    np.testing.assert_array_equal(itd.dx_dt, np.ones((10, 1)))


def test_training_data_x_dxdt_read_only(implicit_training_data):
    x = np.zeros(10)
    dx_dt = np.ones((10, 1))
    itd = implicit_training_data(x, dx_dt)
    _ = itd.x
    _ = itd.dx_dt

    with pytest.raises(AttributeError):
        itd.x = 1
    
    with pytest.raises(AttributeError):
        itd.dx_dt = 1


def test_training_data_slicing(sample_training_data):
    indices = [2, 4, 6, 8]
    sliced_etd = sample_training_data[indices]
    expected_x = np.array([[i * 3, i * 3 + 1, i * 3 + 2] for i in indices])
    expected_dx_dt = np.array([[3, 2, 1]]*len(indices), dtype=float)
    np.testing.assert_array_equal(sliced_etd.x, expected_x)
    np.testing.assert_array_equal(sliced_etd.dx_dt, expected_dx_dt)


@pytest.mark.parametrize("num_elements", range(1, 4))
def test_training_data_len(implicit_training_data, num_elements):
    x = np.arange(num_elements)
    dx_dt = np.arange(num_elements).reshape((-1, 1))
    etd = implicit_training_data(x, dx_dt)
    assert len(etd) == num_elements


def test_correct_partial_calculation_in_training_data(implicit_training_data):
    data_input = np.arange(20, dtype=float).reshape((20, 1))
    data_input = np.c_[data_input * 0,
                       data_input * 1,
                       data_input * 2]
    training_data = implicit_training_data(data_input)

    expected_derivative = np.c_[np.ones(13) * 0,
                                np.ones(13) * 1,
                                np.ones(13) * 2]
    np.testing.assert_array_almost_equal(training_data.dx_dt,
                                         expected_derivative)


def test_correct_partial_calculation_in_training_data_2_sections(
        implicit_training_data):
    data_input = np.arange(20, dtype=float).reshape((20, 1)) * 2.0
    data_input = np.vstack((data_input, [np.nan], data_input))
    training_data = implicit_training_data(data_input)
    expected_derivative = np.full((26, 1), 2.0)
    np.testing.assert_array_almost_equal(training_data.dx_dt,
                                         expected_derivative)


@pytest.mark.parametrize("required_params, expected_fit",
                         [(None, 0.5), (2, 0.5), (3, np.inf)])
def test_implicit_regression(implicit_regression, sample_training_data,
                             sample_equation, required_params, expected_fit):
    if required_params is None:
        reg = implicit_regression(sample_training_data)
    else:
        reg = implicit_regression(sample_training_data, required_params)
    fit_vec = reg.evaluate_fitness_vector(sample_equation)
    expected_fit_vec = np.full((10,), expected_fit, dtype=float)
    np.testing.assert_array_almost_equal(fit_vec, expected_fit_vec)


def test_can_pickle(sample_implicit_regression):
    _ = dill.loads(dill.dumps(sample_implicit_regression))


def test_can_copy(sample_implicit_regression):
    copied = copy(sample_implicit_regression)

    np.testing.assert_array_equal(copied.training_data.x, sample_implicit_regression.training_data.x)
    np.testing.assert_array_equal(copied.training_data.dx_dt, sample_implicit_regression.training_data.dx_dt)
    assert copied.eval_count == sample_implicit_regression.eval_count
