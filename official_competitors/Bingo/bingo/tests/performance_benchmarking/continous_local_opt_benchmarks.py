import timeit

import numpy as np

from bingo.symbolic_regression.agraph \
    import agraph as agraph_module
from bingo.symbolic_regression.agraph.evaluation_backend import \
    evaluation_backend as pyBackend
from bingo.local_optimizers.continuous_local_opt \
    import ContinuousLocalOptimization
from bingocpp import evaluation_backend as cppBackend

from benchmark_data import StatsPrinter, \
                           generate_random_individuals, \
                           copy_to_cpp, \
                           TEST_EXPLICIT_REGRESSION, \
                           TEST_EXPLICIT_REGRESSION_CPP, \
                           TEST_IMPLICIT_REGRESSION, \
                           TEST_IMPLICIT_REGRESSION_CPP, \
                           CLO_TIMING_NUMBER, \
                           CLO_TIMING_REPEATS, \
                           NUM_AGRAPHS_INDVS

import benchmark_data as benchmark_data


TEST_EXPLICIT_REGRESSION_OPTIMIZATION \
    = ContinuousLocalOptimization(TEST_EXPLICIT_REGRESSION)
TEST_IMPLICIT_REGRESSION_OPTIMIZATION \
    = ContinuousLocalOptimization(TEST_IMPLICIT_REGRESSION)
TEST_EXPLICIT_REGRESSION_OPTIMIZATION_CPP \
    = ContinuousLocalOptimization(TEST_EXPLICIT_REGRESSION_CPP)
TEST_IMPLICIT_REGRESSION_OPTIMIZATION_CPP \
    = ContinuousLocalOptimization(TEST_IMPLICIT_REGRESSION_CPP)


TEST_ITERATION = 0
DEBUG = False
TEST_AGRAPHS = generate_random_individuals(benchmark_data.NUM_AGRAPHS_INDVS,
                                           benchmark_data.COMMAND_ARRAY_SIZE,
                                           True)
TEST_AGRAPHS_CPP = copy_to_cpp(TEST_AGRAPHS)
BENCHMARK_LISTS = []
BENCHMARK_LISTS_CPP = []


def benchmark_explicit_regression_with_optimization():
    np.random.seed(0)
    for i, test_run in enumerate(BENCHMARK_LISTS):
        for indv in test_run:
            _ = TEST_EXPLICIT_REGRESSION_OPTIMIZATION.__call__(indv)


def benchmark_explicit_regression_cpp_with_optimization():
    np.random.seed(0)
    for i, test_run in enumerate(BENCHMARK_LISTS_CPP):
        for indv in test_run:
            _ = TEST_EXPLICIT_REGRESSION_OPTIMIZATION_CPP.__call__(indv)


def benchmark_implicit_regression_with_optimization():
    np.random.seed(0)
    for i, test_run in enumerate(BENCHMARK_LISTS):
        for indv in test_run:
            _ = TEST_IMPLICIT_REGRESSION_OPTIMIZATION.__call__(indv)


def benchmark_implicit_regression_cpp_with_optimization():
    np.random.seed(0)
    for i, test_run in enumerate(BENCHMARK_LISTS_CPP):
        for indv in test_run:
            _ = TEST_IMPLICIT_REGRESSION_OPTIMIZATION_CPP.__call__(indv)


def reset_test_data():
    _reset_test_data_helper(TEST_AGRAPHS, BENCHMARK_LISTS)


def reset_test_data_cpp():
    _reset_test_data_helper(TEST_AGRAPHS_CPP, BENCHMARK_LISTS_CPP)


def _reset_test_data_helper(agraph_list, benchmarking_array):
    global TEST_ITERATION
    if DEBUG:
        print("...Executing iteration:", TEST_ITERATION)
    _create_fresh_benchmarking_array(agraph_list, benchmarking_array)
    TEST_ITERATION += 1


def _create_fresh_benchmarking_array(agraph_list, benchmarking_array):
    benchmarking_array.clear()
    for num_runs in range(0, CLO_TIMING_NUMBER):
        run_list = []
        for agraph in agraph_list:
            run_list.append(agraph.copy())
        benchmarking_array.append(run_list)


def do_benchmarking(debug = False):
    if debug:
        global DEBUG
        DEBUG = True

    benchmarks = [
        [benchmark_explicit_regression_with_optimization,
         benchmark_explicit_regression_cpp_with_optimization,
         "LOCAL OPTIMIZATION (EXPLICIT REGRESSION) BENCHMARKS"],
        [benchmark_implicit_regression_with_optimization,
         benchmark_implicit_regression_cpp_with_optimization,
         "LOCAL OPTIMIZATION (IMPLICIT REGRESSION) BENCHMARKS"]]

    stats_printer_list = []
    for regression, regression_cpp, reg_name in benchmarks:
        printer = StatsPrinter(reg_name)
        _run_benchmarks(printer, regression, regression_cpp)
        stats_printer_list.append(printer)

    return stats_printer_list


def _run_benchmarks(printer, regression, regression_cpp):
    for backend, name in [[pyBackend, " py"], [cppBackend, "c++"]]:
        agraph_module.evaluation_backend = backend
        _add_stats_and_log_intermediate_steps(
            printer,
            regression,
            "py:  fitness " +name + ": evaluate ",
            reset_test_data
        )
    _add_stats_and_log_intermediate_steps(
        printer,
        regression_cpp,
        "c++: fitness c++: evaluate ",
        reset_test_data_cpp
    )


def _add_stats_and_log_intermediate_steps(stats_printer,
                                          regression, 
                                          run_name,
                                          setup_function):
    if DEBUG:
        print("Running", regression.__name__, "...................")
    stats_printer.add_stats(
        run_name,
        timeit.repeat(regression, setup=setup_function, number=1,
                      repeat=CLO_TIMING_REPEATS),
        number=CLO_TIMING_NUMBER * NUM_AGRAPHS_INDVS,
        unit_mult=1000
    )
    if DEBUG:
        print(regression.__name__, "finished\n")
    _reset_iteration_count()


def _reset_iteration_count():
    global TEST_ITERATION
    TEST_ITERATION = 0


def _print_stats(printer_list):
    for printer in printer_list:
        printer.print()


if __name__ == '__main__':
    do_benchmarking(False)