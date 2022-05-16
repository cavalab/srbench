# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import pytest

from bingo.chromosomes.multiple_values import MultipleValueChromosome, \
    MultipleValueChromosomeGenerator, SinglePointCrossover, SinglePointMutation

DUMMY_VALUE = 999


@pytest.fixture
def chromosome():
    values = [1, 2, 3]
    chromosome = MultipleValueChromosome(values)
    chromosome.fitness = 0
    return chromosome


@pytest.mark.parametrize("distance", range(4))
def test_chromosome_distance(distance, chromosome):
    values_2 = list(chromosome.values)
    for i in range(distance):
        values_2[i] += DUMMY_VALUE
    chromosome_2 = MultipleValueChromosome(values_2)

    assert chromosome.distance(chromosome_2) == distance
    assert chromosome_2.distance(chromosome) == distance


@pytest.mark.parametrize("values_per_chromosome", range(4))
def test_generator(mocker, values_per_chromosome):
    random_value_function = mocker.Mock()
    generator = MultipleValueChromosomeGenerator(random_value_function,
                                                 values_per_chromosome)
    chromosome = generator()
    assert random_value_function.call_count == values_per_chromosome
    assert len(chromosome.values) == values_per_chromosome


def test_generator_throws_error_for_invalid_chromosome_length(mocker):
    random_value_function = mocker.Mock()
    with pytest.raises(ValueError):
        MultipleValueChromosomeGenerator(random_value_function, -1)


def test_mutation_doesnt_change_parent(chromosome):
    def dummy_function():
        return DUMMY_VALUE
    mutation = SinglePointMutation(dummy_function)
    _ = mutation(chromosome)
    assert DUMMY_VALUE not in chromosome.values


def test_mutation_is_single_point(chromosome):
    def dummy_function():
        return DUMMY_VALUE
    mutation = SinglePointMutation(dummy_function)
    child = mutation(chromosome)
    assert sum([v == DUMMY_VALUE for v in child.values]) == 1


def test_mutation_resets_fitness(chromosome):
    def dummy_function():
        return DUMMY_VALUE
    mutation = SinglePointMutation(dummy_function)
    child = mutation(chromosome)
    assert not child.fit_set


def test_crossover_is_single_point():
    parent_1 = MultipleValueChromosome([1]*10)
    parent_2 = MultipleValueChromosome([2]*10)

    crossover = SinglePointCrossover()

    child_1, child_2 = crossover(parent_1, parent_2)
    ind_1 = child_1.values.index(2)
    ind_2 = child_2.values.index(1)
    assert ind_1 == ind_2
    assert child_1.values[:ind_1] == [1]*ind_1
    assert child_2.values[:ind_2] == [2]*ind_2
    assert child_1.values[ind_1:] == [2]*(10 - ind_1)
    assert child_2.values[ind_2:] == [1]*(10 - ind_2)


def test_crossover_resets_fitness(chromosome):
    crossover = SinglePointCrossover()
    child_1, child_2 = crossover(chromosome, chromosome)

    assert not child_1.fit_set
    assert not child_1.fit_set


def test_crossover_uses_oldest_genetic_age(chromosome):
    older_chromosome = chromosome.copy()
    older_chromosome.genetic_age = 100

    crossover = SinglePointCrossover()
    child_1, child_2 = crossover(chromosome, older_chromosome)
    child_3, child_4 = crossover(older_chromosome, chromosome)

    assert child_1.genetic_age == 100
    assert child_2.genetic_age == 100
    assert child_3.genetic_age == 100
    assert child_4.genetic_age == 100
