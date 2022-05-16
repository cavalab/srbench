"""
An example of bingo genetic optimization used to solve the one max problem.
"""
import numpy as np
from bingo.variation.var_or import VarOr
from bingo.evaluation.fitness_function import FitnessFunction
from bingo.evaluation.evaluation import Evaluation
from bingo.selection.tournament import Tournament
from bingo.evolutionary_algorithms.evolutionary_algorithm \
    import EvolutionaryAlgorithm
from bingo.evolutionary_optimizers.island import Island
from bingo.chromosomes.multiple_values \
    import MultipleValueChromosomeGenerator, SinglePointCrossover, \
    SinglePointMutation

np.random.seed(0)  # used for reproducibility


def run_one_max_problem():
    generator = create_chromosome_generator()
    ev_alg = create_evolutionary_algorithm()

    island = Island(ev_alg, generator, population_size=10)
    display_best_individual(island)

    island.evolve(num_generations=50)

    display_best_individual(island)


def create_chromosome_generator():
    return MultipleValueChromosomeGenerator(generate_0_or_1,
                                            values_per_chromosome=16)


def generate_0_or_1():
    """A function used in generation of values in individuals"""
    return np.random.choice([0, 1])


def create_evolutionary_algorithm():
    crossover = SinglePointCrossover()
    mutation = SinglePointMutation(generate_0_or_1)
    variation_phase = VarOr(crossover, mutation, crossover_probability=0.4,
                            mutation_probability=0.4)

    fitness = OneMaxFitnessFunction()
    evaluation_phase = Evaluation(fitness)

    selection_phase = Tournament(tournament_size=2)

    return EvolutionaryAlgorithm(variation_phase, evaluation_phase,
                                  selection_phase)


class OneMaxFitnessFunction(FitnessFunction):
    """Callable class to calculate fitness"""
    def __call__(self, individual):
        """Fitness = number of 0 elements in the individual's values"""
        self.eval_count += 1
        return individual.values.count(0)


def display_best_individual(island):
    print("Best individual: ", island.get_best_individual())
    print("Best individual's fitness: ", island.get_best_fitness())


if __name__ == "__main__":
    run_one_max_problem()
