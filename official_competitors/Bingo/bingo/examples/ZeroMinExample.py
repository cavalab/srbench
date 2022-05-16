# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import numpy as np

from bingo.evaluation.fitness_function import FitnessFunction
from bingo.evolutionary_algorithms.mu_plus_lambda import MuPlusLambda
from bingo.selection.tournament import Tournament
from bingo.evaluation.evaluation import Evaluation
from bingo.evolutionary_optimizers.island import Island
from bingo.local_optimizers.continuous_local_opt \
    import ContinuousLocalOptimization
from bingo.chromosomes.multiple_values \
    import SinglePointCrossover, SinglePointMutation
from bingo.chromosomes.multiple_floats import MultipleFloatChromosomeGenerator


class ZeroMinFitnessFunction(FitnessFunction):
    def __call__(self, individual):
        return np.linalg.norm(individual.values)


def get_random_float():
    return np.random.random_sample() * 2.


def main():
    crossover = SinglePointCrossover()
    mutation = SinglePointMutation(get_random_float)
    selection = Tournament(10)
    fitness = ZeroMinFitnessFunction()
    local_opt_fitness = ContinuousLocalOptimization(fitness)
    evaluator = Evaluation(local_opt_fitness)
    ea = MuPlusLambda(evaluator, selection, crossover, mutation, 0.4, 0.4, 20)
    generator = MultipleFloatChromosomeGenerator(get_random_float, 8)
    island = Island(ea, generator, 25)

    island.evolve(1)
    report_max_min_mean_fitness(island)
    island.evolve(500)
    report_max_min_mean_fitness(island)


def report_max_min_mean_fitness(island):
    fitness = [indv.fitness for indv in island.population]
    print("Max fitness: \t", np.max(fitness))
    print("Min fitness: \t", np.min(fitness))
    print("Mean fitness: \t", np.mean(fitness))


if __name__ == '__main__':
    main()
