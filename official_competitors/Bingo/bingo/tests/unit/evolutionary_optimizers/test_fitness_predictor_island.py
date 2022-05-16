# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import pytest
from bingo.evolutionary_optimizers.fitness_predictor_island \
    import FitnessPredictorIsland


@pytest.mark.parametrize("param, illegal_value",
                         [("predictor_population_size", -1),
                          ("predictor_update_frequency", 0),
                          ("predictor_size_ratio", 0),
                          ("predictor_size_ratio", 1.2),
                          ("predictor_computation_ratio", -0.2),
                          ("predictor_computation_ratio", 1),
                          ("trainer_population_size", -1),
                          ("trainer_update_frequency", 0)])
def test_raises_error_on_illegal_value_in_init(mocker, param, illegal_value):
    kwargs = {param: illegal_value}
    with pytest.raises(ValueError):
        _ = FitnessPredictorIsland(evolution_algorithm=mocker.Mock(),
                                   generator=mocker.Mock(),
                                   population_size=10, **kwargs)