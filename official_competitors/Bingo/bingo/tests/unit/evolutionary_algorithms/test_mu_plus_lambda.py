# Ignoring some linting rules in tests
# pylint: disable=missing-docstring
import pytest
from bingo.evolutionary_algorithms.mu_plus_lambda import MuPlusLambda
from bingo.evolutionary_algorithms import mu_plus_lambda


@pytest.mark.parametrize("target_pop_size", [None, 10])
def test_all_phases_occur_in_correct_order(mocker, target_pop_size):
    dummy_population = [0]*10
    dummy_offspring = [1]*10
    dummy_next_gen = [2]*10

    mocked_crossover = mocker.Mock()
    mocked_mutation = mocker.Mock()
    mocked_variation = mocker.Mock(return_value=dummy_offspring)
    mocked_evaluation = mocker.Mock()
    mocked_selection = mocker.Mock(return_value=dummy_next_gen)
    mocker.patch("bingo.evolutionary_algorithms."
                 "evolutionary_algorithm.EaDiagnostics", autospec=True)
    mocker.patch("bingo.evolutionary_algorithms."
                 "mu_plus_lambda.VarOr", autospec=True,
                 return_value=mocked_variation)

    evo_alg = MuPlusLambda(mocked_evaluation, mocked_selection,
                           mocked_crossover, mocked_mutation,
                           target_population_size=target_pop_size,
                           crossover_probability=0.5,
                           mutation_probability=0.3,
                           number_offspring=10)
    new_pop = evo_alg.generational_step(dummy_population)

    mocked_variation.assert_called_once()
    assert mocked_evaluation.call_count == 2
    mocked_selection.assert_called_once()

    assert mocked_variation.call_args[0][0] == dummy_population
    assert mocked_evaluation.call_args_list[0][0][0] == \
           dummy_population
    assert mocked_evaluation.call_args_list[1][0][0] == \
           dummy_offspring
    assert mocked_selection.call_args[0][0] == \
           dummy_population + dummy_offspring
    assert new_pop == dummy_next_gen


def test_creates_var_or(mocker):
    mocked_crossover = mocker.Mock()
    mocked_mutation = mocker.Mock()
    mocked_evaluation = mocker.Mock()
    mocked_selection = mocker.Mock()
    mocker.patch("bingo.evolutionary_algorithms."
                 "mu_plus_lambda.VarOr", autospec=True)

    _ = MuPlusLambda(mocked_evaluation, mocked_selection, mocked_crossover,
                     mocked_mutation, crossover_probability=0.5,
                     mutation_probability=0.3, number_offspring=10)

    mu_plus_lambda.VarOr.assert_called_once_with(mocked_crossover,
                                                 mocked_mutation, 0.5, 0.3)
