# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import pytest
import numpy as np

from bingo.symbolic_regression.benchmarking.benchmark_suite import BenchmarkSuite


def test_benchmark_suite_finds_all_benchmark_definitions():
    suite = BenchmarkSuite()
    assert len(suite) == 51


@pytest.mark.parametrize("inclusive_terms, expected_names",
                         [(["Koza"], ["Koza-1", "Koza-2", "Koza-3"]),
                          (["Koza", "1"], ["Koza-1"])])
def test_benchmark_suite_inclusive(inclusive_terms, expected_names):
    suite = BenchmarkSuite(inclusive_terms=inclusive_terms)
    names = [s.name for s in suite]
    assert set(names) == set(expected_names)


def test_benchmark_suite_exclusive():
    suite = BenchmarkSuite(inclusive_terms=["Koza"],
                           exclusive_terms=["2", "3"])
    names = [s.name for s in suite]
    assert set(names) == {"Koza-1"}


def test_benchmark_suite_get_item():
    suite = BenchmarkSuite(inclusive_terms=["Koza-1"])
    assert suite[0].name == "Koza-1"


@pytest.mark.parametrize("repeats", range(1, 10, 3))
@pytest.mark.parametrize("inclusive_term", ["Koza-1", "Koza"])
def test_benchmark_suite_running_test(repeats, inclusive_term, mocker):
    suite = BenchmarkSuite(inclusive_terms=[inclusive_term])

    benchmark_test = mocker.Mock()
    benchmark_test._train_function.return_value = "equ", "aux"
    _, _ = suite.run_benchmark_test(benchmark_test, repeats=repeats)

    num_expected_train = len(suite) * repeats
    num_expected_score = num_expected_train * 2

    assert benchmark_test.train.call_count == num_expected_train
    assert benchmark_test.score.call_count == num_expected_score


@pytest.mark.parametrize("repeats", range(1, 10, 3))
@pytest.mark.parametrize("score_size", range(1, 4))
@pytest.mark.parametrize("inclusive_term", ["Koza-1", "Koza"])
def test_benchmark_suite_running_test_has_correct_output_size(repeats,
                                                              score_size,
                                                              inclusive_term,
                                                              mocker,):
    suite = BenchmarkSuite(inclusive_terms=[inclusive_term])

    benchmark_test = mocker.Mock()
    benchmark_test._train_function.return_value = "equ", "aux"
    benchmark_test.score.return_value = (0,) * score_size
    train_res, test_res = suite.run_benchmark_test(benchmark_test,
                                                   repeats=repeats)
    expected_shape = (len(suite), repeats, score_size)

    assert np.array(train_res).shape == expected_shape
    assert np.array(test_res).shape == expected_shape
