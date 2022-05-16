import timeit
import unittest.mock as mock

import numpy as np

from bingo.symbolic_regression.agraph.crossover import AGraphCrossover
from bingo.symbolic_regression.agraph.mutation import AGraphMutation
from bingo.symbolic_regression.agraph.generator import AGraphGenerator
from bingo.symbolic_regression.agraph.component_generator \
    import ComponentGenerator
from bingo.symbolic_regression.explicit_regression import ExplicitRegression, \
                                                        ExplicitTrainingData
from bingo.evolutionary_algorithms.age_fitness import AgeFitnessEA
from bingo.evaluation.evaluation import Evaluation
from bingo.evolutionary_optimizers.island import Island
from bingo.local_optimizers.continuous_local_opt \
    import ContinuousLocalOptimization
from benchmark_data import StatsPrinter

POP_SIZE = 128
STACK_SIZE = 64
MUTATION_PROBABILITY = 0.4
CROSSOVER_PROBABILITY = 0.4
NUM_POINTS = 100
START = -10
STOP = 10
ERROR_TOLERANCE = 10e-9
SEED = 20


def init_x_vals(start, stop, num_points):
    return np.linspace(start, stop, num_points).reshape([-1, 1])


def equation_eval(x):
    return x**2 + 3.5*x**3


def init_island():
    np.random.seed(15)
    x = init_x_vals(START, STOP, NUM_POINTS)
    y = equation_eval(x)
    training_data = ExplicitTrainingData(x, y)

    component_generator = ComponentGenerator(x.shape[1])
    component_generator.add_operator(2)
    component_generator.add_operator(3)
    component_generator.add_operator(4)

    crossover = AGraphCrossover()
    mutation = AGraphMutation(component_generator)

    agraph_generator = AGraphGenerator(STACK_SIZE, component_generator)

    fitness = ExplicitRegression(training_data=training_data)
    local_opt_fitness = ContinuousLocalOptimization(fitness, algorithm='lm')
    evaluator = Evaluation(local_opt_fitness)

    ea_algorithm = AgeFitnessEA(evaluator, agraph_generator, crossover,
                                mutation, MUTATION_PROBABILITY,
                                CROSSOVER_PROBABILITY, POP_SIZE)

    island = Island(ea_algorithm, agraph_generator, POP_SIZE)
    return island


TEST_ISLAND = init_island()


class IslandStatsPrinter(StatsPrinter):
    def __init__(self):
        super().__init__()
        self._output = ["-"*24+":::: REGRESSION BENCHMARKS ::::" + "-"*23,
                        self._header_format_string.format("NAME", "MEAN",
                                                          "STD", "MIN", "MAX"),
                        "-"*78]


def explicit_regression_benchmark():
    island = init_island()
    while island.get_best_individual().fitness > ERROR_TOLERANCE:
        island._execute_generational_step()


def do_benchmarking():
    printer = IslandStatsPrinter()
    printer.add_stats("Explicit Regression",
                      timeit.repeat(explicit_regression_benchmark,
                                    number=4,
                                    repeat=4))
    printer.print()


if __name__ == "__main__":
    do_benchmarking()
