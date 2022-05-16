# Ignoring some linting rules in tests
# pylint: disable=missing-docstring
import pytest
import numpy as np
from bingo.variation.add_random_individuals import AddRandomIndividuals


@pytest.mark.parametrize("indvs_added", range(1, 5))
def test_random_individuals_added_to_pop(mocker, indvs_added):
    dummy_population = [0]*10
    mocked_variation = mocker.Mock(return_value=dummy_population)
    mocked_variation.crossover_offspring = np.ones(10, dtype=bool)
    mocked_variation.mutation_offspring = np.ones(10, dtype=bool)
    mocked_variation.offspring_parents = [[1]]*10
    mocked_generator = mocker.Mock(return_value=1)

    rand_indv_var_or = AddRandomIndividuals(mocked_variation,
                                            mocked_generator,
                                            num_rand_indvs=indvs_added)
    offspring = rand_indv_var_or(dummy_population, 10)
    assert len(offspring) == 10 + indvs_added
    assert offspring.count(1) == indvs_added


@pytest.mark.parametrize("indvs_added", range(1, 5))
def test_diagnostics(mocker, indvs_added):
    dummy_population = [0]*10
    mocked_variation = mocker.Mock(return_value=dummy_population)
    mocked_variation.crossover_offspring = np.ones(10, dtype=bool)
    mocked_variation.mutation_offspring = np.ones(10, dtype=bool)
    mocked_variation.offspring_parents = [[1]]*10
    mocked_generator = mocker.Mock(return_value=1)

    rand_indv_var_or = AddRandomIndividuals(mocked_variation,
                                            mocked_generator,
                                            num_rand_indvs=indvs_added)
    _ = rand_indv_var_or(dummy_population, 10)

    assert len(rand_indv_var_or.crossover_offspring) == 10 + indvs_added
    assert len(rand_indv_var_or.mutation_offspring) == 10 + indvs_added
    assert len(rand_indv_var_or.offspring_parents) == 10 + indvs_added

    assert not any(rand_indv_var_or.crossover_offspring[-indvs_added:])
    assert not any(rand_indv_var_or.mutation_offspring[-indvs_added:])
    assert rand_indv_var_or.offspring_parents[-indvs_added:] == \
            [[]]*indvs_added