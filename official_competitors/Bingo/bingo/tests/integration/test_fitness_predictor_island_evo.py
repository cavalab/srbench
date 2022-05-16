# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import pytest
import numpy as np

from bingo.chromosomes.multiple_values import SinglePointCrossover, \
    SinglePointMutation, MultipleValueChromosomeGenerator
from bingo.evolutionary_optimizers.fitness_predictor_island \
    import FitnessPredictorIsland as FPI
from bingo.evolutionary_optimizers \
    import fitness_predictor_island as fpi_module
from bingo.evolutionary_algorithms.mu_plus_lambda import MuPlusLambda
from bingo.selection.tournament import Tournament
from bingo.evaluation.evaluation import Evaluation
from bingo.evaluation.fitness_function import FitnessFunction
from bingo.stats.hall_of_fame import HallOfFame


MAIN_POPULATION_SIZE = 40
PREDICTOR_POPULATION_SIZE = 4
TRAINER_POPULATION_SIZE = 4
SUBSET_TRAINING_DATA_SIZE = 2
FULL_TRAINING_DATA_SIZE = 20


class DistanceToAverage(FitnessFunction):
    def __call__(self, individual):
        self.eval_count += 1
        avg_data = np.mean(self.training_data)
        return np.linalg.norm(individual.values - avg_data)


@pytest.fixture
def full_training_data():
    return np.linspace(0.1, 1, FULL_TRAINING_DATA_SIZE)


@pytest.fixture
def ev_alg(full_training_data):
    crossover = SinglePointCrossover()
    mutation = SinglePointMutation(np.random.random)
    selection = Tournament(2)
    fitness = DistanceToAverage(full_training_data)
    evaluator = Evaluation(fitness)
    return MuPlusLambda(evaluator, selection, crossover, mutation,
                        0., 1.0, MAIN_POPULATION_SIZE)


@pytest.fixture
def generator():
    return MultipleValueChromosomeGenerator(np.random.random, 10)


@pytest.fixture
def fitness_predictor_island(ev_alg, generator):
    island = FPI(ev_alg, generator, MAIN_POPULATION_SIZE,
                 predictor_population_size=PREDICTOR_POPULATION_SIZE,
                 trainer_population_size=TRAINER_POPULATION_SIZE,
                 predictor_size_ratio=SUBSET_TRAINING_DATA_SIZE/FULL_TRAINING_DATA_SIZE,
                 predictor_computation_ratio=0.4,
                 trainer_update_frequency=4,
                 predictor_update_frequency=5)
    island._predictor_island._ea.variation._mutation_probability = 1.0
    return island


@pytest.fixture
def fp_island_and_hof(ev_alg, generator):
    hof = HallOfFame(5)
    fp_island = FPI(ev_alg, generator, MAIN_POPULATION_SIZE,
                    predictor_population_size=PREDICTOR_POPULATION_SIZE,
                    trainer_population_size=TRAINER_POPULATION_SIZE,
                    predictor_size_ratio=SUBSET_TRAINING_DATA_SIZE/FULL_TRAINING_DATA_SIZE,
                    predictor_computation_ratio=0.4,
                    trainer_update_frequency=4,
                    predictor_update_frequency=5,
                    hall_of_fame=hof)
    fp_island._predictor_island._ea.variation._mutation_probability = 1.0
    return fp_island, hof


def test_best_fitness_is_true_fitness(fitness_predictor_island,
                                      full_training_data):

    true_fitness_function = DistanceToAverage(full_training_data)
    best_individual = fitness_predictor_island.get_best_individual()
    best_fitness = fitness_predictor_island.get_best_fitness()
    expected_best_fitness = true_fitness_function(best_individual)
    assert best_fitness == expected_best_fitness


def test_predictor_compute_ratios(fitness_predictor_island):
    # init
    point_evals_predictor = FULL_TRAINING_DATA_SIZE*TRAINER_POPULATION_SIZE
    point_evals_predictor += 2 * point_evals_per_predictor_step()
    point_evals_main = 0
    assert_expected_compute_ratio(fitness_predictor_island,
                                  point_evals_main, point_evals_predictor)

    # main step
    fitness_predictor_island.evolve(1, suppress_logging=True)
    point_evals_main += 2 * point_evals_per_main_step()
    assert_expected_compute_ratio(fitness_predictor_island,
                                  point_evals_main, point_evals_predictor)

    # main + predictor
    fitness_predictor_island.evolve(1, suppress_logging=True)
    point_evals_main += point_evals_per_main_step()
    point_evals_predictor += point_evals_per_predictor_step()
    assert_expected_compute_ratio(fitness_predictor_island,
                                  point_evals_main, point_evals_predictor)

    # main + 2 predictor
    fitness_predictor_island.evolve(1, suppress_logging=True)
    point_evals_main += point_evals_per_main_step()
    point_evals_predictor += 2 * point_evals_per_predictor_step()
    assert_expected_compute_ratio(fitness_predictor_island,
                                  point_evals_main, point_evals_predictor)

    # main + predictor + trainer update
    fitness_predictor_island.evolve(1, suppress_logging=True)
    point_evals_main += point_evals_per_main_step()
    point_evals_predictor += point_evals_per_predictor_step()
    point_evals_predictor += point_evals_per_trainer_update()
    assert_expected_compute_ratio(fitness_predictor_island,
                                  point_evals_main, point_evals_predictor)

    # main + predictor update
    fitness_predictor_island.evolve(1, suppress_logging=True)
    point_evals_main += point_evals_per_main_step()
    point_evals_main += point_evals_per_predictor_update()
    assert_expected_compute_ratio(fitness_predictor_island,
                                  point_evals_main, point_evals_predictor)


def test_fitness_predictor_island_ages(fitness_predictor_island):
    predictor_age = 1
    main_age = 0
    assert fitness_predictor_island.generational_age == main_age
    assert fitness_predictor_island._predictor_island.generational_age \
        == predictor_age

    fitness_predictor_island._execute_generational_step()
    main_age += 1
    assert fitness_predictor_island.generational_age == main_age
    assert fitness_predictor_island._predictor_island.generational_age \
        == predictor_age

    fitness_predictor_island._execute_generational_step()
    main_age += 1
    predictor_age += 1
    assert fitness_predictor_island.generational_age == main_age
    assert fitness_predictor_island._predictor_island.generational_age \
        == predictor_age

    fitness_predictor_island._execute_generational_step()
    main_age += 1
    predictor_age += 2
    assert fitness_predictor_island.generational_age == main_age
    assert fitness_predictor_island._predictor_island.generational_age \
        == predictor_age

    fitness_predictor_island._execute_generational_step()
    main_age += 1
    predictor_age += 1
    assert fitness_predictor_island.generational_age == main_age
    assert fitness_predictor_island._predictor_island.generational_age \
        == predictor_age


def test_nan_on_predicted_variance_of_trainer(mocker,
                                              fitness_predictor_island):
    mocker.patch('bingo.evolutionary_optimizers.'
                 'fitness_predictor_island.np.var')
    fpi_module.np.var.side_effect = OverflowError

    island = fitness_predictor_island
    trainer = island.population[0]
    variance = island._calculate_predictor_variance_of(trainer)
    assert np.isnan(variance)


def test_hof_gets_filled(fp_island_and_hof):
    fp_island, hof = fp_island_and_hof
    fp_island.evolve(1)
    assert len(hof) == 5


def test_hof_has_true_fitness(fp_island_and_hof, full_training_data):
    fp_island, hof = fp_island_and_hof
    true_fitness_function = DistanceToAverage(full_training_data)

    fp_island.evolve(1)

    for indv in hof:
        true_fitness = true_fitness_function(indv)
        assert indv.fitness == pytest.approx(true_fitness)


def test_temp_hof_is_cleared_with_predictor_update(fp_island_and_hof, mocker):
    fp_island, hof = fp_island_and_hof
    mocker.spy(fp_island._hof_w_predicted_fitness, 'clear')
    fp_island.evolve(9)
    assert fp_island._hof_w_predicted_fitness.clear.call_count == 1


def assert_expected_compute_ratio(fitness_predictor_island, point_evals_main,
                                  point_evals_predictor):
    current_ratio = \
        fitness_predictor_island._get_predictor_computation_ratio()
    np.testing.assert_almost_equal(current_ratio,
                                   point_evals_predictor /
                                   (point_evals_predictor + point_evals_main))


def point_evals_per_predictor_step():
    return SUBSET_TRAINING_DATA_SIZE * PREDICTOR_POPULATION_SIZE \
           * TRAINER_POPULATION_SIZE


def point_evals_per_main_step():
    return SUBSET_TRAINING_DATA_SIZE * MAIN_POPULATION_SIZE


def point_evals_per_trainer_update():
    return SUBSET_TRAINING_DATA_SIZE * MAIN_POPULATION_SIZE * \
           PREDICTOR_POPULATION_SIZE + FULL_TRAINING_DATA_SIZE + \
           point_evals_per_predictor_step()


def point_evals_per_predictor_update():
    return point_evals_per_main_step()
