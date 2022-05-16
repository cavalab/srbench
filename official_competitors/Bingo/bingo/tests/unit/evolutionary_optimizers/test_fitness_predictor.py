# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import pytest
import numpy as np
from bingo.evolutionary_optimizers.fitness_predictor\
    import FitnessPredictorFitnessFunction, FitnessPredictorIndexGenerator
from bingo.chromosomes.multiple_values import MultipleValueChromosome


@pytest.fixture
def mocked_training_data(mocker):
    return mocker.MagicMock()


@pytest.fixture
def mocked_population(mocker):
    return [mocker.Mock() for _ in range(10)]


@pytest.fixture
def mocked_fitness_function(mocker):
    return mocker.Mock(return_value=1.0)


@pytest.fixture
def predictor_fit_func(mocked_training_data, mocked_population,
                       mocked_fitness_function):
    return FitnessPredictorFitnessFunction(mocked_training_data,
                                           mocked_fitness_function,
                                           mocked_population,
                                           num_trainers=5)


def test_raises_error_invalid_num_trainers(mocked_training_data,
                                           mocked_population,
                                           mocked_fitness_function):
    with pytest.raises(ValueError):
        _ = FitnessPredictorFitnessFunction(mocked_training_data,
                                            mocked_fitness_function,
                                            mocked_population,
                                            num_trainers=-1)


def test_raises_error_not_enough_trainers(mocked_training_data,
                                          mocked_population,
                                          mocked_fitness_function):
    with pytest.raises(RuntimeError):
        _ = FitnessPredictorFitnessFunction(mocked_training_data,
                                            mocked_fitness_function,
                                            mocked_population,
                                            num_trainers=11)


def test_raises_error_not_enough_valid_trainers(mocker, mocked_training_data,
                                                mocked_population):
    mocked_nan_fitness_function = mocker.Mock(return_value=np.nan)
    with pytest.raises(RuntimeError):
        _ = FitnessPredictorFitnessFunction(mocked_training_data,
                                            mocked_nan_fitness_function,
                                            mocked_population,
                                            num_trainers=5)


def test_adding_trainers_to_predictor_fitness_function(mocker,
                                                       predictor_fit_func):
    for i in range(10):
        trainer = mocker.Mock()
        trainer.copy.return_value = i
        predictor_fit_func.add_trainer(trainer)
        for j in range(max(0, i-4), i + 1):
            assert j in predictor_fit_func._trainers


@pytest.mark.parametrize("maximum", [2, 20])
def test_index_generator(maximum):
    generator = FitnessPredictorIndexGenerator(maximum)
    indices = np.array([generator() for _ in range(100)])
    assert np.all(indices >= 0)
    assert np.all(indices < maximum)


def test_fitness_predictor_fitness_function_call(mocker):
    mocked_training_data = mocker.MagicMock()
    train_fit = [2.0, 0.0]
    mocked_fitness_function = mocker.Mock(side_effect=train_fit)
    mocked_population = [mocker.Mock() for _ in range(5)]
    predictor_fitness_function = \
        FitnessPredictorFitnessFunction(mocked_training_data,
                                        mocked_fitness_function,
                                        mocked_population,
                                        num_trainers=2)
    pred_fit = 1.5
    mocker.patch.object(predictor_fitness_function,
                        "predict_fitness_for_trainer",
                        return_value=pred_fit)
    predictor = mocker.Mock()
    fitness = predictor_fitness_function(predictor)
    assert fitness == pytest.approx(1.0)


def test_predicted_fitness_for_trainer(mocker, predictor_fit_func):
    mocked_predictor = mocker.create_autospec(MultipleValueChromosome)
    mocked_predictor.values = [1, 2, 3]
    mocked_trainer = mocker.Mock()

    predictor_fit_func.predict_fitness_for_trainer(mocked_predictor,
                                                   mocked_trainer)
    predictor_fit_func.training_data.__getitem__.assert_called_once_with(
            [1, 2, 3])
    assert predictor_fit_func._fitness_function.call_args[0][0] ==\
        mocked_trainer


def test_get_true_fitness_for_trainer(mocker, predictor_fit_func):
    mocked_trainer = mocker.Mock()
    predictor_fit_func.get_true_fitness_for_trainer(mocked_trainer)
    assert predictor_fit_func._fitness_function.training_data == \
        predictor_fit_func.training_data
    assert predictor_fit_func._fitness_function.call_args[0][0] == \
        mocked_trainer
