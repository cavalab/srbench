# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import csv
import numpy as np

from bingo.symbolic_regression.agraph.generator import AGraphGenerator
from bingo.symbolic_regression.agraph.component_generator \
    import ComponentGenerator
from bingo.symbolic_regression.implicit_regression \
    import ImplicitRegression, ImplicitTrainingData, _calculate_partials
from bingo.symbolic_regression.explicit_regression \
    import ExplicitRegression, ExplicitTrainingData
import bingocpp

LOG_WIDTH = 78
NUM_AGRAPHS_INDVS = 100
COMMAND_ARRAY_SIZE = 128
NUM_X_VALUES = 128

EVAL_TIMING_NUMBER = 50
EVAL_TIMING_REPEATS = 10

FITNESS_TIMING_NUMBER = 50
FITNESS_TIMING_REPEATS = 10

CLO_TIMING_NUMBER = 4
CLO_TIMING_REPEATS = 4


class StatsPrinter:
    def __init__(self, title="PERFORMANCE BENCHMARKS"):
        self._header_format_string = \
            "{:<26}   {:>10} +- {:<10}   {:^10}   {:^10}"
        self._format_string = \
            "{:<26}   {:>10.4f} +- {:<10.4f}   {:^10.4f}   {:^10.4f}"
        diff = LOG_WIDTH - len(title) - 10
        self._output = [
            "-"*int(diff/2)+":::: {} ::::".format(title) + "-"*int((diff + 1)/2),
            self._header_format_string.format("NAME", "MEAN",
                                              "STD", "MIN", "MAX"),
            "-"*LOG_WIDTH]

    def add_stats(self, name, times, number=1, unit_mult=1):
        std_time = np.std(times) / number * unit_mult
        mean_time = np.mean(times) / number * unit_mult
        max_time = np.max(times) / number * unit_mult
        min_time = np.min(times) / number * unit_mult

        self._output.append(self._format_string.format(name, mean_time,
                                                       std_time, min_time,
                                                       max_time))

    def print(self):
        for line in self._output:
            print(line)
        print()


def generate_random_individuals(num_individuals, stack_size,
                                optimize_constants=False):
    np.random.seed(0)
    generate_agraph = set_up_agraph_generator(stack_size)

    individuals = generate_indv_list_that_needs_local_optimiziation(
        generate_agraph, num_individuals)
    if not optimize_constants:
        set_constants(individuals)

    return individuals


def set_up_agraph_generator(stack_size):
    generator = ComponentGenerator(input_x_dimension=4,
                                   num_initial_load_statements=2,
                                   terminal_probability=0.1)
    for i in range(2, 13):
        generator.add_operator(i)
    generate_agraph = AGraphGenerator(stack_size, generator, use_python=True)
    return generate_agraph


def generate_indv_list_that_needs_local_optimiziation(generate_agraph,
                                                      num_individuals):
    count = 0
    indv_list = []
    while count < num_individuals:
        indv = generate_agraph()
        if indv.needs_local_optimization():
            indv_list.append(indv)
            count += 1
    return indv_list


def set_constants(individuals):
    for indv in individuals:
        num_consts = indv.get_number_local_optimization_params()
        if num_consts > 0:
            consts = np.random.rand(num_consts) * 10.0
            indv.set_local_optimization_params(consts)


def copy_to_cpp(indvs_python):
    indvs_cpp = []
    for indv in indvs_python:
        agraph_cpp = bingocpp.AGraph()
        agraph_cpp.genetic_age = indv.genetic_age
        agraph_cpp.fitness = indv.fitness if indv.fitness is not None else 1e9
        agraph_cpp.fit_set = indv.fit_set
        agraph_cpp.set_local_optimization_params(indv.constants)
        agraph_cpp.command_array = indv.command_array
        indvs_cpp.append(agraph_cpp)
    return indvs_cpp


def generate_random_x(size):
    np.random.seed(0)
    return np.random.rand(size, 4)*10 - 5.0


def write_stacks(test_agraph_list):
    filename = '../bingocpp/app/test-agraph-stacks.csv'
    with open(filename, mode='w+') as stack_file:
        stack_file_writer = csv.writer(stack_file, delimiter=',')
        for agraph in test_agraph_list:
            stack = []
            for row in agraph.command_array:
                for i in np.nditer(row):
                    stack.append(i)
            stack_file_writer.writerow(stack)
    stack_file.close()


def write_constants(test_agraph_list):
    filename = '../bingocpp/app/test-agraph-consts.csv'
    with open(filename, mode='w+') as const_file:
        const_file_writer = csv.writer(const_file, delimiter=',')
        for agraph in test_agraph_list:
            consts = agraph.constants
            num_consts = len(consts)
            consts = np.insert(consts, 0, num_consts, axis=0)
            const_file_writer.writerow(consts)
    const_file.close()


def write_x_vals(test_x_vals):
    filename = '../bingocpp/app/test-agraph-x-vals.csv'
    with open(filename, mode='w+') as x_file:
        x_file_writer = csv.writer(x_file, delimiter=',')
        for row in test_x_vals:
            x_file_writer.writerow(row)
    x_file.close()


def initialize_implicit_data(initial_x):
    x, dx_dt, _ = _calculate_partials(initial_x)
    return x, dx_dt


def explicit_regression():
    training_data = ExplicitTrainingData(TEST_X_PARTIALS, TEST_Y_ZEROS)
    return ExplicitRegression(training_data)


def explicit_regression_cpp():
    training_data = bingocpp.ExplicitTrainingData(TEST_X_PARTIALS,
                                                  TEST_Y_ZEROS)
    return bingocpp.ExplicitRegression(training_data)


def implicit_regression():
    training_data = ImplicitTrainingData(TEST_X_PARTIALS, TEST_DX_DT)
    return ImplicitRegression(training_data)


def implicit_regression_cpp():
    training_data = bingocpp.ImplicitTrainingData(TEST_X_PARTIALS,
                                                  TEST_DX_DT)
    return bingocpp.ImplicitRegression(training_data)


TEST_X = generate_random_x(NUM_X_VALUES)
TEST_X_PARTIALS, TEST_DX_DT = initialize_implicit_data(TEST_X)
TEST_Y_ZEROS = np.zeros(TEST_X_PARTIALS.shape[0]).reshape((-1, 1))

TEST_AGRAPHS = generate_random_individuals(NUM_AGRAPHS_INDVS,
                                           COMMAND_ARRAY_SIZE)
TEST_AGRAPHS_CPP = copy_to_cpp(TEST_AGRAPHS)

TEST_EXPLICIT_REGRESSION = explicit_regression()
TEST_EXPLICIT_REGRESSION_CPP = explicit_regression_cpp()
TEST_IMPLICIT_REGRESSION = implicit_regression()
TEST_IMPLICIT_REGRESSION_CPP = implicit_regression_cpp()
