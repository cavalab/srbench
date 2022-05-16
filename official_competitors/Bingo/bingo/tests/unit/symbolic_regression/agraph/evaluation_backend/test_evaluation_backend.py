# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import numpy as np
import pytest

from bingo.symbolic_regression.agraph.operator_definitions import *
from bingo.symbolic_regression.agraph.evaluation_backend \
    import evaluation_backend as py_eval_backend

try:
    from bingocpp import evaluation_backend as cpp_eval_backend
except ImportError:
    cpp_eval_backend = None

CPP_PARAM = pytest.param("Cpp",
                         marks=pytest.mark.skipif(not cpp_eval_backend,
                                                  reason='BingoCpp import '
                                                         'failure'))

OPERATOR_LIST = [INTEGER, VARIABLE, CONSTANT, ADDITION, SUBTRACTION,
                 MULTIPLICATION, DIVISION, SIN, COS, EXPONENTIAL, LOGARITHM,
                 POWER, ABS, SQRT]


@pytest.fixture(params=["Python", CPP_PARAM])
def engine(request):
    return request.param


@pytest.fixture
def eval_backend(engine):
    if engine == "Python":
        return py_eval_backend
    return cpp_eval_backend


@pytest.fixture
def all_funcs_command_array():
    return np.array([[INTEGER, 5, 5],
                     [VARIABLE, 0, 0],
                     [CONSTANT, 0, 0],
                     [ADDITION, 1, 0],
                     [SUBTRACTION, 2, 3],
                     [MULTIPLICATION, 4, 1],
                     [DIVISION, 5, 1],
                     [SIN, 6, 0],
                     [COS, 7, 0],
                     [EXPONENTIAL, 8, 0],
                     [LOGARITHM, 9, 0],
                     [SAFE_POWER, 10, 0],
                     [ABS, 11, 0],
                     [SQRT, 12, 0]])


@pytest.fixture
def higher_dim_command_array():
    return np.array([[VARIABLE, 0, 0],
                     [VARIABLE, 1, 1],
                     [CONSTANT, 0, 0],
                     [CONSTANT, 1, 1],
                     [MULTIPLICATION, 0, 2],
                     [MULTIPLICATION, 1, 3],
                     [ADDITION, 4, 5]])


@pytest.fixture
def sample_x():
    return np.vstack((np.linspace(-1.0, 0.0, 11),
                      np.linspace(0.0, 1.0, 11))).transpose()


@pytest.fixture
def sample_constants():
    return np.array([10, 3.14])


def test_all_funcs_eval(eval_backend, all_funcs_command_array):
    x = np.arange(1, 6).reshape((-1, 1))
    constants = (10, )
    expected_f_of_x = np.array([[0.45070097],
                                [0.9753327],
                                [0.29576841],
                                [0.36247937],
                                [1.0]])
    f_of_x = eval_backend.evaluate(all_funcs_command_array,
                                   x, constants)
    np.testing.assert_array_almost_equal(f_of_x, expected_f_of_x)


def test_higher_dim_func_eval(eval_backend, higher_dim_command_array):
    x = np.arange(8).reshape((-1, 2))
    constants = (10, 100)
    expected_f_of_x = np.sum(x*constants, axis=1).reshape((-1, 1))
    f_of_x = eval_backend.evaluate(higher_dim_command_array,
                                   x, constants)
    np.testing.assert_array_almost_equal(f_of_x, expected_f_of_x)


def test_all_funcs_deriv_x(eval_backend, all_funcs_command_array):
    x = np.arange(1, 6).reshape((-1, 1))
    constants = (10, )
    expected_f_of_x = np.array([[0.45070097],
                                [0.9753327],
                                [0.29576841],
                                [0.36247937],
                                [1.0]])
    expected_df_dx = np.array([[0.69553357],
                               [-0.34293336],
                               [-0.39525239],
                               [0.54785643],
                               [0.0]])
    f_of_x, df_dx = eval_backend.evaluate_with_derivative(
            all_funcs_command_array, x, constants, True)
    np.testing.assert_array_almost_equal(f_of_x, expected_f_of_x)
    np.testing.assert_array_almost_equal(df_dx, expected_df_dx)


def test_all_funcs_deriv_c(eval_backend, all_funcs_command_array):
    x = np.arange(1, 6).reshape((-1, 1))
    constants = (10, )
    expected_f_of_x = np.array([[0.45070097],
                                [0.9753327],
                                [0.29576841],
                                [0.36247937],
                                [1.0]])
    expected_df_dc = np.array([[-0.69553357],
                               [0.34293336],
                               [0.39525239],
                               [-0.54785643],
                               [0.]])
    f_of_x, df_dc = eval_backend.evaluate_with_derivative(
            all_funcs_command_array, x, constants, False)
    np.testing.assert_array_almost_equal(f_of_x, expected_f_of_x)
    np.testing.assert_array_almost_equal(df_dc, expected_df_dc)


def test_higher_dim_func_deriv_x(eval_backend, higher_dim_command_array):
    x = np.arange(8).reshape((4, 2))
    constants = (10, 100)
    expected_f_of_x = np.sum(x*constants, axis=1).reshape((-1, 1))
    expected_df_dx = np.array([constants]*4)

    f_of_x, df_dx = eval_backend.evaluate_with_derivative(
            higher_dim_command_array, x, constants, True)
    np.testing.assert_array_almost_equal(f_of_x, expected_f_of_x)
    np.testing.assert_array_almost_equal(df_dx, expected_df_dx)


def test_higher_dim_func_deriv_c(eval_backend, higher_dim_command_array):
    x = np.arange(8).reshape((4, 2))
    constants = (10, 100)
    expected_f_of_x = np.sum(x*constants, axis=1).reshape((-1, 1))
    expected_df_dc = x

    f_of_x, df_dc = eval_backend.evaluate_with_derivative(
            higher_dim_command_array, x, constants, False)
    np.testing.assert_array_almost_equal(f_of_x, expected_f_of_x)
    np.testing.assert_array_almost_equal(df_dc, expected_df_dc)

