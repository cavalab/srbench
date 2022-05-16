# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import pytest
import numpy as np

from bingo.chromosomes.multiple_values import SinglePointCrossover, \
    SinglePointMutation, MultipleValueChromosomeGenerator
from bingo.evolutionary_optimizers.island import Island
from bingo.evolutionary_algorithms.mu_plus_lambda import MuPlusLambda
from bingo.selection.tournament import Tournament
from bingo.evaluation.evaluation import Evaluation
from bingo.evaluation.fitness_function import FitnessFunction


class MultipleValueFitnessFunction(FitnessFunction):
    def __call__(self, individual):
        fitness = np.count_nonzero(individual.values)
        self.eval_count += 1
        return len(individual.values) - fitness


@pytest.fixture
def island():
    crossover = SinglePointCrossover()
    mutation = SinglePointMutation(mutation_function)
    selection = Tournament(10)
    fitness = MultipleValueFitnessFunction()
    evaluator = Evaluation(fitness)
    ev_alg = MuPlusLambda(evaluator, selection, crossover, mutation,
                          0.2, 0.4, 20)
    generator = MultipleValueChromosomeGenerator(mutation_function, 10)
    return Island(ev_alg, generator, 25)


def mutation_function():
    return np.random.choice([True, False])


def test_manual_evaluation(island):
    island.evaluate_population()
    for indv in island.population:
        assert indv.fit_set


def test_generational_steps_change_population_age(island):
    for indv in island.population:
        assert indv.genetic_age == 0
    island._execute_generational_step()
    for indv in island.population:
        assert indv.genetic_age > 0


def test_generational_age_increases(island):
    island.evolve(1)
    assert island.generational_age == 1
    island.evolve(1)
    assert island.generational_age == 2
    island.evolve(10)
    assert island.generational_age == 12


def test_best_individual(island):
    island.evolve(1)
    fitness = [indv.fitness for indv in island.population]
    best = island.get_best_individual()
    assert best.fitness == min(fitness)


def test_best_fitness(island):
    island.evolve(1)
    fitness = [indv.fitness for indv in island.population]
    best_fitness = island.get_best_fitness()
    assert best_fitness == min(fitness)


def test_best_evaluation_count(island):
    assert island.get_fitness_evaluation_count() == 0
    island.evolve(1)
    assert island.get_fitness_evaluation_count() == 45


def test_island_hof(mocker):
    hof = mocker.Mock()
    crossover = SinglePointCrossover()
    mutation = SinglePointMutation(mutation_function)
    selection = Tournament(10)
    fitness = MultipleValueFitnessFunction()
    evaluator = Evaluation(fitness)
    ev_alg = MuPlusLambda(evaluator, selection, crossover, mutation,
                          0.2, 0.4, 20)
    generator = MultipleValueChromosomeGenerator(mutation_function, 10)
    island = Island(ev_alg, generator, 25, hall_of_fame=hof)

    island.evolve(10)

    hof.update.assert_called_once()
    hof_update_pop = hof.update.call_args[0][0]
    for h, i in zip(hof_update_pop, island.population):
        assert h == i
