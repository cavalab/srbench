import timeit

from bingo.symbolic_regression.agraph \
    import agraph as agraph_module
from bingo.symbolic_regression.agraph.evaluation_backend import \
    evaluation_backend as pyBackend
from bingocpp import evaluation_backend as cppBackend

import benchmark_data as benchmark_data
from benchmark_data import TEST_AGRAPHS, TEST_X, EVAL_TIMING_NUMBER, \
                           EVAL_TIMING_REPEATS, NUM_AGRAPHS_INDVS

def benchmark_evaluate():
    for indv in TEST_AGRAPHS:
        _ = indv.evaluate_equation_at(TEST_X)


def benchmark_evaluate_w_x_derivative():
    for indv in TEST_AGRAPHS:
        _ = indv.evaluate_equation_with_x_gradient_at(TEST_X)


def benchmark_evaluate_w_c_derivative():
    for indv in TEST_AGRAPHS:
        _ = indv.evaluate_equation_with_local_opt_gradient_at(TEST_X)


def do_benchmarking():
    printer = benchmark_data.StatsPrinter("EVALUATION BENCHMARKS")
    for backend, name in [[pyBackend, "py"], [cppBackend, "c++"]]:
        agraph_module.evaluation_backend = backend
        printer.add_stats(name + ": evaluate",
                          timeit.repeat(benchmark_evaluate,
                                        number=EVAL_TIMING_NUMBER,
                                        repeat=EVAL_TIMING_REPEATS),
                          number=EVAL_TIMING_NUMBER * NUM_AGRAPHS_INDVS,
                          unit_mult=1000)
        printer.add_stats(name + ": x derivative",
                          timeit.repeat(benchmark_evaluate_w_x_derivative,
                                        number=EVAL_TIMING_NUMBER,
                                        repeat=EVAL_TIMING_REPEATS),
                          number=EVAL_TIMING_NUMBER * NUM_AGRAPHS_INDVS,
                          unit_mult=1000)
        printer.add_stats(name + ": c derivative",
                          timeit.repeat(benchmark_evaluate_w_c_derivative,
                                        number=EVAL_TIMING_NUMBER,
                                        repeat=EVAL_TIMING_REPEATS),
                          number=EVAL_TIMING_NUMBER * NUM_AGRAPHS_INDVS,
                          unit_mult=1000)
    return printer