# Ignoring some linting rules in tests
# pylint: disable=missing-docstring
# pylint: disable=redefined-outer-name
from bingo.evolutionary_algorithms.evolutionary_algorithm \
    import EvolutionaryAlgorithm


def test_all_phases_occur_in_ea(mocker):
    dummy_population = [0]*10
    dummy_offspring = [1]*10
    dummy_next_gen = [2]*10
    mocked_variation = mocker.Mock(return_value=dummy_offspring)
    mocked_evaluation = mocker.Mock()
    mocked_selection = mocker.Mock(return_value=dummy_next_gen)
    mocker.patch("bingo.evolutionary_algorithms."
                 "evolutionary_algorithm.EaDiagnostics", autospec=True)

    evo_alg = EvolutionaryAlgorithm(mocked_variation, mocked_evaluation,
                                    mocked_selection)
    new_pop = evo_alg.generational_step(dummy_population)

    mocked_variation.assert_called_once()
    assert mocked_evaluation.call_count == 2
    mocked_selection.assert_called_once()

    assert mocked_variation.call_args[0][0] == dummy_population
    assert mocked_evaluation.call_args_list[0][0][0] == \
           dummy_population
    assert mocked_evaluation.call_args_list[1][0][0] == \
           dummy_offspring
    assert mocked_selection.call_args[0][0] == dummy_offspring
    assert new_pop == dummy_next_gen