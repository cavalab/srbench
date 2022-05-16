# Ignoring some linting rules in tests
# pylint: disable=missing-docstring
import pytest
import numpy as np
from bingo.chromosomes.chromosome import Chromosome
from bingo.selection.age_fitness import AgeFitness


@pytest.fixture
def dummy_chromosome(mocker):
    mocker.patch('bingo.chromosomes.chromosome.Chromosome', autospec=True)
    mocker.patch.object(Chromosome, "__abstractmethods__",
                        new_callable=set)
    return Chromosome


@pytest.fixture
def population_01234(dummy_chromosome):
    return [dummy_chromosome(fitness=i, genetic_age=i) for i in range(5)]


@pytest.fixture
def population_nans(dummy_chromosome, population_01234):
    return [dummy_chromosome(fitness=np.nan) for _ in range(5)] + \
           population_01234


@pytest.fixture
def three_non_dominated_population(dummy_chromosome):
    return [dummy_chromosome(fitness=i, genetic_age=j)
            for i, j in [(0, 2), (1, 1), (2, 0), (0, 3), (3, 0), (1, 2)]]

@pytest.fixture
def non_dominated_population(dummy_chromosome):
    return [dummy_chromosome(fitness=i, genetic_age=6 - i) for i in range(6)]


@pytest.mark.parametrize("selection_size,expected_error", [
    (0, ValueError),
    ("string", TypeError)
])
def test_raises_error_invalid_tournament_size(selection_size, expected_error):
    with pytest.raises(expected_error):
        _ = AgeFitness(selection_size)


def test_raises_error_with_too_large_target_population(population_01234):
    age_fitness = AgeFitness()
    with pytest.raises(ValueError):
        _ = age_fitness(population_01234, target_population_size=6)


@pytest.mark.parametrize("selection_size", [2, 5])
def test_return_initial_population_if_non_dominated(non_dominated_population,
                                                    selection_size):
    age_fitness = AgeFitness(selection_size=selection_size)
    new_pop = age_fitness(non_dominated_population, target_population_size=2)
    assert new_pop == non_dominated_population


@pytest.mark.parametrize("target_size", range(1, 5))
def test_target_population_size(population_01234, target_size):
    age_fitness = AgeFitness()
    new_pop = age_fitness(population_01234, target_size)
    assert len(new_pop) == target_size


@pytest.mark.parametrize("target_size", [2, 5])
@pytest.mark.parametrize("selection_size", [2, 5])
def test_target_population_size_with_nans(population_nans, target_size,
                                          selection_size):
    age_fitness = AgeFitness(selection_size=selection_size)
    new_pop = age_fitness(population_nans, target_size)
    assert len(new_pop) == target_size


@pytest.mark.parametrize("selection_size", [2, 5])
def test_proper_selection(three_non_dominated_population, selection_size):
    age_fitness = AgeFitness(selection_size=selection_size)
    new_pop = age_fitness(three_non_dominated_population,
                          target_population_size=2)
    new_pop = sorted(new_pop, key=lambda x: x.fitness)
    assert new_pop == three_non_dominated_population[:3]
