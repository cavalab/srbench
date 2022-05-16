# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import pytest
import numpy as np

from bingo.symbolic_regression.benchmarking.benchmark \
    import Benchmark, AnalyticBenchmark


def sample_eval_func(x):
    return (np.sum(x, axis=1) + 1).reshape((-1, 1))


def test_benchmark_uses_explicit_training_data():
    a = Benchmark("name", "description", "source", np.array([[1]]),
                  np.array([[2]]), np.array([[3]]), np.array([[4]]))
    np.testing.assert_array_almost_equal(a.training_data.x, np.array([[1]]))
    np.testing.assert_array_almost_equal(a.training_data.y, np.array([[2]]))
    np.testing.assert_array_almost_equal(a.test_data.x, np.array([[3]]))
    np.testing.assert_array_almost_equal(a.test_data.y, np.array([[4]]))


def test_analytic_benchmark_even_distribution():
    dist = ("E", 1, 3, 1)
    a = AnalyticBenchmark("name", "description", "source", 2, sample_eval_func,
                          training_x_distribution=dist,
                          test_x_distribution=dist)
    np.testing.assert_array_almost_equal(a.training_data.x,
                                         np.array([[1, 1], [2, 2], [3, 3]]))


def test_analytic_benchmark_uniform_distribution():
    dist = ("U", 1, 3, 3)
    a = AnalyticBenchmark("name", "description", "source", 2, sample_eval_func,
                          training_x_distribution=dist,
                          test_x_distribution=dist)
    np.testing.assert_array_almost_equal(a.training_data.x,
                                         np.array([[2.097627, 2.089766],
                                                   [2.430379, 1.84731 ],
                                                   [2.205527, 2.291788]]))


def test_analytic_benchmark_multiple_distributions():
    dist = [("U", 1, 3, 3), ("E", 1, 3, 1)]
    a = AnalyticBenchmark("name", "description", "source", 2, sample_eval_func,
                          training_x_distribution=dist,
                          test_x_distribution=dist)
    np.testing.assert_array_almost_equal(a.training_data.x,
                                         np.array([[2.097627, 1],
                                                   [2.430379, 2],
                                                   [2.205527, 3]]))


@pytest.mark.parametrize("dist, expected_error",
                         [([("U", 1, 3, 3), ("U", 0, 2, 3)], RuntimeError),
                          (("X", 1, 3, 3), KeyError)])
def test_analytic_benchmark_invalid_distributions(dist, expected_error):
    with pytest.raises(expected_error):
        _ = AnalyticBenchmark("name", "description", "source", 1,
                              sample_eval_func, dist, dist)


def test_analytic_benchmark_propper_eval():
    dist = ("E", 1, 3, 1)
    a = AnalyticBenchmark("name", "description", "source", 2, sample_eval_func,
                          training_x_distribution=dist,
                          test_x_distribution=dist)
    np.testing.assert_array_almost_equal(a.training_data.y,
                                         np.array([[3], [5], [7]]))


def test_analytic_benchmark_doesnt_set_consistent_random_seed():
    dist = ("U", 1, 3, 3)
    _ = AnalyticBenchmark("name", "description", "source", 2, sample_eval_func,
                          dist, dist)
    first_rand = np.random.random((2, 2))
    _ = AnalyticBenchmark("name", "description", "source", 2, sample_eval_func,
                          dist, dist)
    second_rand = np.random.random((2, 2))

    assert np.linalg.norm(first_rand - second_rand) > 1e-6
