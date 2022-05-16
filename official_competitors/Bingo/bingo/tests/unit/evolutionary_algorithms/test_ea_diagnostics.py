# Ignoring some linting rules in tests
# pylint: disable=missing-docstring

import pytest
import numpy as np
from bingo.chromosomes.chromosome import Chromosome
from bingo.evolutionary_algorithms.ea_diagnostics \
    import EaDiagnostics, EaDiagnosticsSummary


@pytest.fixture
def dummy_chromosome(mocker):
    mocker.patch('bingo.chromosomes.chromosome.Chromosome', autospec=True)
    mocker.patch.object(Chromosome, "__abstractmethods__",
                        new_callable=set)
    return Chromosome


@pytest.fixture
def population_12(dummy_chromosome):
    return [dummy_chromosome(fitness=i) for i in [1, 2]]


@pytest.fixture
def population_0123_times_4(dummy_chromosome):
    return [dummy_chromosome(fitness=i) for i in [0, 1, 2, 3] * 4]


def test_correctly_updated_summary(population_12, population_0123_times_4):
    offspring_parents = [[0, 1]] * 8 + [[0]] * 6 + [[]]*2
    offspring_crossover = np.array([1] * 8 + [0] * 8, dtype=bool)
    offspring_mutation = np.array([0] * 4 + [1] * 8 + [0] * 4, dtype=bool)

    ead = EaDiagnostics()
    ead.update(population_12, population_0123_times_4, offspring_parents,
               offspring_crossover, offspring_mutation)

    expected_summary = EaDiagnosticsSummary(
            beneficial_crossover_rate=0.25,
            detrimental_crossover_rate=0.25,
            beneficial_mutation_rate=0.25,
            detrimental_mutation_rate=0.5,
            beneficial_crossover_mutation_rate=0.25,
            detrimental_crossover_mutation_rate=0.25)

    assert ead.summary == expected_summary


@pytest.mark.parametrize("num_subsets", [1, 2, 4, 8])
def test_sum(population_12, population_0123_times_4, num_subsets):
    offspring_parents = [[0, 1]] * 8 + [[0]] * 8
    offspring_crossover = np.array([1] * 8 + [0] * 8, dtype=bool)
    offspring_mutation = np.array([0] * 4 + [1] * 8 + [0] * 4, dtype=bool)

    num_subsets = 2
    ead_list = []
    for i in range(num_subsets):
        subset_inds = list(range(i, 16, num_subsets))
        offspring = [population_0123_times_4[i] for i in subset_inds]
        parents = [offspring_parents[i] for i in subset_inds]
        cross = offspring_crossover[subset_inds]
        mut = offspring_mutation[subset_inds]
        ead = EaDiagnostics()
        ead.update(population_12, offspring, parents, cross, mut)
        ead_list.append(ead)

    expected_summary = EaDiagnosticsSummary(
            beneficial_crossover_rate=0.25,
            detrimental_crossover_rate=0.25,
            beneficial_mutation_rate=0.25,
            detrimental_mutation_rate=0.5,
            beneficial_crossover_mutation_rate=0.25,
            detrimental_crossover_mutation_rate=0.25)

    assert sum(ead_list).summary == expected_summary
