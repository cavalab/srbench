# Ignoring some linting rules in tests
# pylint: disable=missing-docstring
import pytest
import numpy as np
from bingo.chromosomes.chromosome import Chromosome
from bingo.variation.var_and import VarAnd


@pytest.fixture
def dummy_chromosome(mocker):
    mocker.patch('bingo.chromosomes.chromosome.Chromosome', autospec=True)
    mocker.patch.object(Chromosome, "__abstractmethods__",
                        new_callable=set)
    return Chromosome


@pytest.fixture
def dummy_population(dummy_chromosome):
    return [dummy_chromosome(fitness="replication") for _ in range(5)]


@pytest.fixture
def dummy_crossover(mocker, dummy_chromosome):
    crossover_cromosome = dummy_chromosome(fitness="crossover")
    return mocker.Mock(return_value=(crossover_cromosome,
                                     crossover_cromosome))


@pytest.fixture
def dummy_mutation(mocker, dummy_chromosome):
    mutation_chromosome = dummy_chromosome(fitness="mutation")
    return mocker.Mock(return_value=mutation_chromosome)


@pytest.mark.parametrize("cx_prob, mut_prob",
                         [(-0.1, 0.6),
                          (1.1, 0.6),
                          (0.6, -0.1),
                          (0.6, 1.1)])
def test_invalid_probabilities(mocker, cx_prob, mut_prob):
    crossover = mocker.Mock()
    mutation = mocker.Mock()
    with pytest.raises(ValueError):
        _ = VarAnd(crossover, mutation, cx_prob, mut_prob)


def test_probabilities_are_about_right(dummy_population, dummy_crossover,
                                       dummy_mutation):
    np.random.seed(1)
    variation = VarAnd(dummy_crossover, dummy_mutation, 0.5, 0.3)
    _ = variation(dummy_population, 1000)

    assert np.count_nonzero(variation.crossover_offspring) == 470
    assert np.count_nonzero(variation.mutation_offspring) == 296


def test_diagnostics_source(dummy_population, dummy_crossover, dummy_mutation):
    variation = VarAnd(dummy_crossover, dummy_mutation, 0.5, 0.3)
    offspring = variation(dummy_population, 100)

    for off, cross, mut in zip(offspring, variation.crossover_offspring,
                               variation.mutation_offspring):
        if off.fitness == "replication":
            assert not cross and not mut
        elif off.fitness == "mutation":
            assert mut
        elif off.fitness == "crossover":
            assert cross


def test_diagnostics_parents(dummy_population, dummy_crossover,
                             dummy_mutation):
    variation = VarAnd(dummy_crossover, dummy_mutation, 0.5, 0.3)
    _ = variation(dummy_population, 100)

    for parents, cross, mut in zip(variation.offspring_parents,
                                   variation.crossover_offspring,
                                   variation.mutation_offspring):
        if cross:
            assert len(parents) == 2
        else:
            assert len(parents) == 1
        assert max(parents) < 5 and min(parents) >= 0


def test_offspring_is_not_parent(dummy_population, dummy_crossover,
                                 dummy_mutation):
    variation = VarAnd(dummy_crossover, dummy_mutation, 0., 0.)
    offspring = variation(dummy_population, 25)

    for indv in enumerate(offspring):
        assert indv not in offspring
