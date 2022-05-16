# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
from collections import namedtuple
import pytest
import numpy as np

from bingo.symbolic_regression.agraph.operator_definitions \
    import VARIABLE, CONSTANT, SIN, ADDITION, SUBTRACTION
from bingo.symbolic_regression.agraph.agraph \
    import AGraph as pyagraph, force_use_of_python_backends

try:
    from bingocpp import AGraph as cppagraph
except ImportError:
    cppagraph = None

force_use_of_python_backends()

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
def sample_agraph(agraph_implementation):  # sin(X_0 + 2.0) + 2.0
    test_graph = agraph_implementation()
    test_graph.command_array = np.array([[VARIABLE, 0, 0],
                                         [CONSTANT, 0, 0],
                                         [ADDITION, 0, 1],
                                         [SIN, 2, 2],
                                         [ADDITION, 0, 1],
                                         [ADDITION, 3, 1]], dtype=int)
    _ = test_graph.needs_local_optimization()
    test_graph.set_local_optimization_params([2.0, ])
    _ = test_graph.mutable_command_array
    return test_graph


@pytest.fixture
def overcomplex_agraph_with_simplification(agraph_implementation):
    sample = agraph_implementation(use_simplification=True)
    sample.command_array = np.array([[VARIABLE, 0, 0],
                                     [SUBTRACTION, 0, 0],
                                     [ADDITION, 1, 0]], dtype=int)
    return sample


@pytest.fixture
def overcomplex_agraph_without_simplification(agraph_implementation):
    sample = agraph_implementation(use_simplification=False)
    sample.command_array = np.array([[VARIABLE, 0, 0],
                                     [SUBTRACTION, 0, 0],
                                     [ADDITION, 1, 0]], dtype=int)
    return sample


@pytest.fixture
def sample_agraph_values():
    values = namedtuple('Data', ['x', 'f_of_x', 'grad_x', 'grad_c'])
    x = np.vstack((np.linspace(-1.0, 0.0, 11),
                   np.linspace(0.0, 1.0, 11))).transpose()
    f_of_x = (np.sin(x[:, 0] + 2.0) + 2.0).reshape((-1, 1))
    grad_x = np.zeros(x.shape)
    grad_x[:, 0] = np.cos(x[:, 0] + 2.0)
    grad_c = (np.cos(x[:, 0] + 2.0) + 1.0).reshape((-1, 1))
    return values(x, f_of_x, grad_x, grad_c)


def test_evaluate_agraph(sample_agraph, sample_agraph_values):
    np.testing.assert_allclose(
        sample_agraph.evaluate_equation_at(sample_agraph_values.x),
        sample_agraph_values.f_of_x)


def test_evaluate_agraph_x_gradient(sample_agraph, sample_agraph_values):
    f_of_x, df_dx = \
        sample_agraph.evaluate_equation_with_x_gradient_at(
            sample_agraph_values.x)
    np.testing.assert_allclose(f_of_x, sample_agraph_values.f_of_x)
    np.testing.assert_allclose(df_dx, sample_agraph_values.grad_x)


def test_evaluate_agraph_c_gradient(sample_agraph, sample_agraph_values):
    f_of_x, df_dc = \
        sample_agraph.evaluate_equation_with_local_opt_gradient_at(
            sample_agraph_values.x)
    np.testing.assert_allclose(f_of_x, sample_agraph_values.f_of_x)
    np.testing.assert_allclose(df_dc, sample_agraph_values.grad_c)


def test_using_simplification(overcomplex_agraph_with_simplification, engine):
    if engine == "c++":
        pytest.xfail(reason="Simplification not yet implemented in c++")

    assert overcomplex_agraph_with_simplification.get_complexity() == 1


def test_not_using_simplification(overcomplex_agraph_without_simplification):
    assert overcomplex_agraph_without_simplification.get_complexity() == 3


def test_evaluate_overflow_exception(mocker, engine, sample_agraph,
                                     sample_agraph_values):
    if engine == "Python":
        mocker.patch("bingo.symbolic_regression.agraph.agraph."
                     "evaluation_backend.evaluate", side_effect=OverflowError)

        values = sample_agraph.evaluate_equation_at(sample_agraph_values.x)
        assert np.isnan(values).all()


def test_evaluate_gradient_overflow_exception(mocker, engine, sample_agraph,
                                              sample_agraph_values):
    if engine == "Python":
        mocker.patch("bingo.symbolic_regression.agraph.agraph."
                     "evaluation_backend.evaluate_with_derivative",
                     side_effect=OverflowError)

        values = sample_agraph.evaluate_equation_with_x_gradient_at(
            sample_agraph_values.x)
        assert np.isnan(values).all()


def test_evaluate_local_opt_gradient_overflow_exception(mocker, engine,
                                                        sample_agraph,
                                                        sample_agraph_values):
    if engine == "Python":
        mocker.patch("bingo.symbolic_regression.agraph.agraph."
                     "evaluation_backend.evaluate_with_derivative",
                     side_effect=OverflowError)

        values = sample_agraph.evaluate_equation_with_local_opt_gradient_at(
                sample_agraph_values.x)
        assert np.isnan(values).all()
