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
                 POWER, ABS, SQRT, SAFE_POWER, SINH, COSH]


@pytest.fixture(params=["Python", CPP_PARAM])
def engine(request):
    return request.param


@pytest.fixture
def eval_backend(engine):
    if engine == "Python":
        return py_eval_backend
    return cpp_eval_backend


@pytest.fixture
def sample_x():
    return np.array([[-2, 1.5],
                     [-1, 0.5],
                     [-0.5, 0],
                     [0, 0],
                     [0, -1],
                     [0.25, -2]])


@pytest.fixture
def sample_constants():
    return np.array([10, 3.14])


def _terminal_evaluations(terminal, x, constants):
    # assumes parameter is 0
    if terminal == INTEGER:
        return np.zeros((x.shape[0], 1))
    if terminal == VARIABLE:
        return x[:, 0].reshape((-1, 1))
    if terminal == CONSTANT:
        return np.full((x.shape[0], 1), constants[0])
    raise NotImplementedError("No test for terminal: %d" % terminal)


def _terminal_derivatives(terminal, deriv_wrt, shape):
    # assumes parameter is 0
    zero = np.zeros(shape)
    one = np.ones(shape)
    if terminal == INTEGER:
        return zero
    if terminal == VARIABLE:
        if deriv_wrt == VARIABLE:
            return one
        return zero
    if terminal == CONSTANT:
        if deriv_wrt == CONSTANT:
            return one
        return zero
    raise NotImplementedError("No test for terminal: %d" % terminal)


def _function_evaluations(function, a, b):
    # returns: f(a,b)
    # or if the function takes only a single parameter: f(a)
    if function == ADDITION:
        return a + b
    if function == SUBTRACTION:
        return a - b
    if function == MULTIPLICATION:
        return a * b
    if function == DIVISION:
        return a / b
    if function == SIN:
        return np.sin(a)
    if function == COS:
        return np.cos(a)
    if function == EXPONENTIAL:
        return np.exp(a)
    if function == LOGARITHM:
        return np.log(np.abs(a))
    if function == POWER:
        return np.power(a, b)
    if function == ABS:
        return np.abs(a)
    if function == SQRT:
        return np.sqrt(np.abs(a))
    if function == SAFE_POWER:
        return np.power(np.abs(a), b)
    if function == SINH:
        return np.sinh(a)
    if function == COSH:
        return np.cosh(a)
    raise NotImplementedError("No test for operator: %d" % function)


def _function_derivatives(function, a, b, da, db):
    # returns: df(a,b)/da, df(a,b)/db
    # or if the function takes only a single parameter: df(a)/da, 0
    zero = np.zeros_like(b)
    if function == ADDITION:
        return da, db
    if function == SUBTRACTION:
        return da, -db
    if function == MULTIPLICATION:
        return da * b, a * db
    if function == DIVISION:
        return da / b, -a * db / b**2
    if function == SIN:
        return da*np.cos(a), zero
    if function == COS:
        return -da*np.sin(a), zero
    if function == EXPONENTIAL:
        return da*np.exp(a), zero
    if function == LOGARITHM:
        return da / a, zero
    if function == POWER:
        return np.power(a, b) * da * b / a, \
               np.power(a, b) * db * np.log(a)
    if function == SAFE_POWER:
        return np.power(np.abs(a), b) * da * b / a, \
               np.power(np.abs(a), b) * db * np.log(np.abs(a))
    if function == ABS:
        return da * np.sign(a), zero
    if function == SQRT:
        return da * np.sign(a) * 0.5 / np.sqrt(np.abs(a)), zero
    if function == SINH:
        return da*np.cosh(a), zero
    if function == COSH:
        return da*np.sinh(a), zero
    raise NotImplementedError("No test for operator: %d" % function)


@pytest.mark.parametrize("operator", OPERATOR_LIST)
def test_operator_evaluate(eval_backend, sample_x, sample_constants, operator):
    if IS_TERMINAL_MAP[operator]:
        expected_outcome = _terminal_evaluations(operator, sample_x,
                                                 sample_constants)
    else:
        x_0 = sample_x[:, 0].reshape((-1, 1))
        x_1 = sample_x[:, 1].reshape((-1, 1))
        expected_outcome = _function_evaluations(operator, x_0, x_1)

    stack = np.array([[VARIABLE, 0, 0],
                      [VARIABLE, 1, 1],
                      [operator, 0, 1]])
    f_of_x = eval_backend.evaluate(stack, sample_x, sample_constants)
    np.testing.assert_allclose(expected_outcome, f_of_x)


@pytest.mark.parametrize("operator", OPERATOR_LIST)
def test_operator_derivative_x0x1(eval_backend, sample_x, sample_constants,
                                  operator):
    expected_outcome = np.zeros_like(sample_x)
    if IS_TERMINAL_MAP[operator]:
        deriv = _terminal_derivatives(operator, deriv_wrt=VARIABLE,
                                      shape=sample_x.shape[0])
        expected_outcome[:, 0] = deriv
    else:
        x_0 = sample_x[:, 0]
        x_1 = sample_x[:, 1]
        dx_0 = np.ones_like(x_0)
        dx_1 = np.ones_like(x_1)
        deriv_0, deriv_1 = _function_derivatives(operator, x_0, x_1,
                                                 dx_0, dx_1)
        expected_outcome[:, 0] = deriv_0
        expected_outcome[:, 1] = deriv_1

    stack = np.array([[VARIABLE, 0, 0],
                      [VARIABLE, 1, 1],
                      [operator, 0, 1]])
    _, df_dx = eval_backend.evaluate_with_derivative(stack, sample_x,
                                                     sample_constants, True)
    np.testing.assert_allclose(expected_outcome, df_dx)


@pytest.mark.parametrize("operator", OPERATOR_LIST)
def test_operator_derivative_x0x0(eval_backend, sample_x, sample_constants,
                                  operator):
    expected_outcome = np.zeros_like(sample_x)
    if IS_TERMINAL_MAP[operator]:
        deriv = _terminal_derivatives(operator, deriv_wrt=VARIABLE,
                                      shape=sample_x.shape[0])
        expected_outcome[:, 0] = deriv
    else:
        x_0 = sample_x[:, 0]
        dx_0 = np.ones_like(x_0)
        deriv_0, deriv_1 = _function_derivatives(operator, x_0, x_0,
                                                 dx_0, dx_0)
        expected_outcome[:, 0] = deriv_0 + deriv_1

    stack = np.array([[VARIABLE, 0, 0],
                      [VARIABLE, 1, 1],
                      [operator, 0, 0]])
    _, df_dx = eval_backend.evaluate_with_derivative(stack, sample_x,
                                                     sample_constants, True)
    np.testing.assert_allclose(expected_outcome, df_dx)


@pytest.mark.parametrize("operator", OPERATOR_LIST)
def test_operator_derivative_c0c1(eval_backend, sample_x, sample_constants,
                                  operator):
    expected_outcome = np.zeros_like(sample_x)
    if IS_TERMINAL_MAP[operator]:
        deriv = _terminal_derivatives(operator, deriv_wrt=CONSTANT,
                                      shape=sample_x.shape[0])
        expected_outcome[:, 0] = deriv
    else:
        c_0 = np.full(sample_x.shape[0], sample_constants[0], dtype=float)
        c_1 = np.full_like(c_0, sample_constants[1], dtype=float)
        dc_0 = np.ones_like(c_0)
        dc_1 = np.ones_like(c_0)
        deriv_0, deriv_1 = _function_derivatives(operator, c_0, c_1,
                                                 dc_0, dc_1)
        expected_outcome[:, 0] = deriv_0
        expected_outcome[:, 1] = deriv_1

    stack = np.array([[CONSTANT, 0, 0],
                      [CONSTANT, 1, 1],
                      [operator, 0, 1]])
    _, df_dc = eval_backend.evaluate_with_derivative(stack, sample_x,
                                                     sample_constants, False)
    np.testing.assert_allclose(expected_outcome, df_dc)


@pytest.mark.parametrize("operator", OPERATOR_LIST)
def test_operator_derivative_with_chain_rule(eval_backend, sample_x,
                                             sample_constants, operator):
    if IS_TERMINAL_MAP[operator]:
        return

    expected_outcome = np.zeros_like(sample_x)
    cx_0 = sample_constants[0] * sample_x[:, 0]
    cx_1 = sample_constants[1] * sample_x[:, 1]
    dcx_0 = np.full_like(cx_0, sample_constants[0], dtype=float)
    dcx_1 = np.full_like(cx_1, sample_constants[1], dtype=float)
    deriv_0, deriv_1 = _function_derivatives(operator, cx_0, cx_1,
                                             dcx_0, dcx_1)
    expected_outcome[:, 0] = deriv_0
    expected_outcome[:, 1] = deriv_1

    stack = np.array([[CONSTANT, 0, 0],
                      [CONSTANT, 1, 1],
                      [VARIABLE, 0, 0],
                      [VARIABLE, 1, 1],
                      [MULTIPLICATION, 0, 2],
                      [MULTIPLICATION, 3, 1],
                      [operator, 4, 5]])
    f, df_dx = eval_backend.evaluate_with_derivative(stack, sample_x,
                                                     sample_constants, True)
    np.testing.assert_allclose(expected_outcome, df_dx)
