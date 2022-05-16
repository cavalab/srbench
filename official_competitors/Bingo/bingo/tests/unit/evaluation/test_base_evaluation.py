# Ignoring some linting rules in tests
# pylint: disable=missing-docstring

from bingo.evaluation.evaluation import Evaluation


def test_evaluation_has_accessor_to_fitness_function_eval_count(mocker):
    mocked_fit_function = mocker.Mock()
    mocked_fit_function.eval_count = 10
    evaluation = Evaluation(mocked_fit_function)

    assert evaluation.eval_count == 10

    evaluation.eval_count = 100
    assert mocked_fit_function.eval_count == 100


def test_evaluation_finds_fitness_for_individuals_that_need_it(mocker):
    population = [mocker.Mock() for _ in range(10)]
    for i, indv in enumerate(population):
        if i in [2, 4, 6, 8]:
            indv.fit_set = False
        else:
            indv.fit_set = True

    mocked_fit_function = mocker.Mock()
    evaluation = Evaluation(mocked_fit_function)
    evaluation(population)

    fit_func_calls = mocked_fit_function.call_args_list
    fit_func_args = [i.call_list()[0][0] for i in fit_func_calls]
    for i, indv in enumerate(population):
        if i in [2, 4, 6, 8]:
            assert (indv, ) in fit_func_args
        else:
            assert (indv, ) not in fit_func_args
