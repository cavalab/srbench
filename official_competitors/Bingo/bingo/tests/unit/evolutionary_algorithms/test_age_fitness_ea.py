# Ignoring some linting rules in tests
# pylint: disable=missing-docstring
from bingo.evolutionary_algorithms.age_fitness \
    import AgeFitnessEA
from bingo.evolutionary_algorithms import age_fitness


def test_creates_var_or(mocker):
    mocked_crossover = mocker.Mock()
    mocked_mutation = mocker.Mock()
    mocked_evaluation = mocker.Mock()
    mocked_generator = mocker.Mock()
    mocked_variation = mocker.Mock()
    mocker.patch("bingo.evolutionary_algorithms."
                 "age_fitness.VarAnd", autospec=True,
                 return_value=mocked_variation)
    mocker.patch("bingo.evolutionary_algorithms."
                 "age_fitness.AgeFitness", autospec=True)
    mocker.patch("bingo.evolutionary_algorithms."
                 "age_fitness.AddRandomIndividuals", autospec=True)

    _ = AgeFitnessEA(mocked_evaluation, mocked_generator, mocked_crossover,
                     mocked_mutation, crossover_probability=0.5,
                     mutation_probability=0.3, population_size=10,
                     selection_size=3)

    age_fitness.VarAnd.assert_called_once_with(mocked_crossover,
                                               mocked_mutation, 0.5, 0.3)
    age_fitness.AgeFitness.assert_called_once_with(3)
    age_fitness.AddRandomIndividuals.assert_called_once_with(mocked_variation,
                                                             mocked_generator)
