# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import pytest

from bingo.chromosomes.multiple_floats import MultipleFloatChromosome,\
                                              MultipleFloatChromosomeGenerator

DUMMY_VALUE = 999


def test_multiple_float_needs_local_optimization():
    chromosome_with_opt = MultipleFloatChromosome([1, 2, 3], [1])
    chromosome_without_opt = MultipleFloatChromosome([1, 2, 3])

    assert chromosome_with_opt. needs_local_optimization()
    assert not chromosome_without_opt. needs_local_optimization()


@pytest.mark.parametrize("num_params", range(3))
def test_getting_number_of_optimization_params(num_params):
    needs_opt_list = list(range(num_params))
    chromosome = MultipleFloatChromosome([1, 2, 3], needs_opt_list)
    assert chromosome.get_number_local_optimization_params() == num_params


def test_setting_optimization_params():
    needs_opt_list = [1, 3, 5]
    chromosome = MultipleFloatChromosome([0] * 6, needs_opt_list)
    chromosome.set_local_optimization_params([1, 1, 1])
    for i in needs_opt_list:
        assert chromosome.values[i] == 1


@pytest.mark.parametrize("bad_opt_list", [[-1, 0],
                                          [0, 4],
                                          [0, 0.5]])
def test_generator_errors_with_bad_opt_list(mocker, bad_opt_list):
    with pytest.raises(ValueError):
        _ = MultipleFloatChromosomeGenerator(mocker.Mock(),
                                             values_per_chromosome=4,
                                             needs_opt_list=bad_opt_list)


def test_generator():
    def dummy_function():
        return DUMMY_VALUE
    needs_opt_list = [1, 3, 5]
    generator = MultipleFloatChromosomeGenerator(dummy_function,
                                                 values_per_chromosome=6,
                                                 needs_opt_list=needs_opt_list)
    chromosome = generator()
    chromosome.set_local_optimization_params([1, 1, 1])
    for i in needs_opt_list:
        assert chromosome.values[i] == 1


def test_generator_default():
    def dummy_function():
        return DUMMY_VALUE
    generator = MultipleFloatChromosomeGenerator(dummy_function,
                                                 values_per_chromosome=6)
    chromosome = generator()
    assert chromosome.get_number_local_optimization_params() == 0

