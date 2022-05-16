# Ignoring some linting rules in tests
# pylint: disable=missing-docstring
# pylint: disable=redefined-outer-name
from bingo.evolutionary_algorithms.deterministic_crowding \
    import DeterministicCrowdingEA
from bingo.evolutionary_algorithms import deterministic_crowding


def test_all_phases_occur_in_correct_order(mocker):
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
                 "deterministic_crowding.VarAnd", autospec=True,
                 return_value=mocked_variation)
    mocker.patch("bingo.evolutionary_algorithms."
                 "deterministic_crowding.DeterministicCrowding", autospec=True,
                 return_value=mocked_selection)

    evo_alg = DeterministicCrowdingEA(mocked_evaluation, mocked_crossover,
                                      mocked_mutation,
                                      crossover_probability=0.5,
                                      mutation_probability=0.3)
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
    mocker.patch("bingo.evolutionary_algorithms."
                 "deterministic_crowding.VarAnd", autospec=True)
    mocker.patch("bingo.evolutionary_algorithms."
                 "deterministic_crowding.DeterministicCrowding", autospec=True)

    _ = DeterministicCrowdingEA(mocked_evaluation, mocked_crossover,
                                mocked_mutation, crossover_probability=0.5,
                                mutation_probability=0.3)

    deterministic_crowding.VarAnd.assert_called_once_with(mocked_crossover,
                                                          mocked_mutation,
                                                          0.5, 0.3)
    deterministic_crowding.DeterministicCrowding.assert_called_once()
