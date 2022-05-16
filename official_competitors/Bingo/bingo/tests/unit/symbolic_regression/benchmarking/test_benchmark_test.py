# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring

import pytest

from bingo.symbolic_regression.benchmarking.benchmark_test import BenchmarkTest
from bingo.symbolic_regression.agraph.agraph import AGraph


def dummy_train_function(training_data):
    dummy_train_graph = AGraph()
    dummy_train_graph.fitness = 1.5
    return dummy_train_graph, None


def dummy_metric_function(equation, data, aux_info):
    return equation.get_complexity(), equation.fitness


@pytest.fixture
def sample_bench_t():
    return BenchmarkTest(train_function=dummy_train_function,
                         score_function=dummy_metric_function)


def test_benchmark_test_train_runs_funct(mocker):
    mock_train_function = mocker.Mock()
    mock_train_function.return_value = ("equ", "aux")
    bench_t = BenchmarkTest(train_function=mock_train_function,
                            score_function=dummy_metric_function)
    bench_t.train("data")
    mock_train_function.assert_called_once_with("data")


def test_benchmark_test_score_runs_funct(mocker):
    def dummy_train(_):
        return "best_equ", "aux_info"
    mock_score_function = mocker.Mock()
    bench_t = BenchmarkTest(train_function=dummy_train,
                            score_function=mock_score_function)
    bench_t.train("train_data")
    bench_t.score("test_data")
    mock_score_function.assert_called_once_with("best_equ", "test_data",
                                                "aux_info")


def test_benchmark_test_score(sample_bench_t):
    sample_bench_t.train("train_data")
    score = sample_bench_t.score("test_data")
    assert score == (0, 1.5)


def test_benchmark_test_must_be_trained_before_test(sample_bench_t):
    with pytest.raises(RuntimeError):
        _ = sample_bench_t.score("")
