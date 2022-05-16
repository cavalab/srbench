import numpy as np

from bingo.symbolic_regression.benchmarking.benchmark_suite \
    import BenchmarkSuite
from bingo.symbolic_regression.benchmarking.benchmark_test \
    import BenchmarkTest
from bingo.symbolic_regression import ComponentGenerator, \
                                      AGraphGenerator, \
                                      AGraphCrossover, \
                                      AGraphMutation, \
                                      ExplicitRegression
from bingo.local_optimizers.continuous_local_opt \
    import ContinuousLocalOptimization
from bingo.evaluation.evaluation import Evaluation
from bingo.evolutionary_algorithms.age_fitness import AgeFitnessEA
from bingo.evolutionary_algorithms.deterministic_crowding \
    import DeterministicCrowdingEA
from bingo.evolutionary_optimizers.island import Island


def training_function(training_data, ea_choice):
    component_generator = \
        ComponentGenerator(input_x_dimension=training_data.x.shape[1])
    component_generator.add_operator("+")
    component_generator.add_operator("-")
    component_generator.add_operator("*")

    agraph_generator = AGraphGenerator(agraph_size=32,
                                       component_generator=component_generator)

    crossover = AGraphCrossover()
    mutation = AGraphMutation(component_generator)

    fitness = ExplicitRegression(training_data=training_data)
    local_opt_fitness = ContinuousLocalOptimization(fitness, algorithm='lm')
    evaluator = Evaluation(local_opt_fitness)

    POPULATION_SIZE = 64
    MUTATION_PROBABILITY = 0.1
    CROSSOVER_PROBABILITY = 0.7

    if ea_choice == "age_fitness":
        ea = AgeFitnessEA(evaluator, agraph_generator, crossover, mutation,
                          MUTATION_PROBABILITY, CROSSOVER_PROBABILITY,
                          POPULATION_SIZE)
    else:
        ea = DeterministicCrowdingEA(evaluator, crossover, mutation,
                                     MUTATION_PROBABILITY,
                                     CROSSOVER_PROBABILITY)

    island = Island(ea, agraph_generator, POPULATION_SIZE)
    opt_result = island.evolve_until_convergence(max_generations=MAX_GENERATIONS,
                                                 fitness_threshold=1e-6)

    return island.get_best_individual(), opt_result


def scoring_function(equation, scoring_data, opt_result):
    mae_function = ExplicitRegression(training_data=scoring_data)
    mae = mae_function(equation)
    return mae, opt_result.success


def parse_results(train_results, test_results):
    train_array = np.array(train_results)
    test_array = np.array(test_results)
    mae_train = np.mean(train_array, axis=1)[:, 0]
    mae_test = np.mean(test_array, axis=1)[:, 0]
    success_rate = np.mean(train_array, axis=1)[:, 1]
    return mae_train, mae_test, success_rate


def print_results(title, af_res, dc_res, bench_names):
    print("\n----------::", title, "::-------------")
    titles = "".join(["{:^10}".format(name) for name in bench_names])
    print("              " + titles)
    af_scores = "".join(["{:^10.2e}".format(score) for score in af_res])
    print("age-fitness   " + af_scores)
    dc_scores = "".join(["{:^10.2e}".format(score) for score in dc_res])
    print("det. crowding " + dc_scores)


def run_benchmark_comparison():
    suite = BenchmarkSuite(inclusive_terms=["Nguyen"])
    age_fitness_strategy = \
        BenchmarkTest(lambda x: training_function(x, "age_fitness"),
                      scoring_function)
    deterministic_crowding_strategy = \
        BenchmarkTest(lambda x: training_function(x, "deterministic_crowding"),
                      scoring_function)

    train_scores_af, test_scores_af = \
        suite.run_benchmark_test(age_fitness_strategy, repeats=NUM_REPEATS)
    train_scores_dc, test_scores_dc = \
        suite.run_benchmark_test(deterministic_crowding_strategy,
                                 repeats=NUM_REPEATS)

    mae_train_af, mae_test_af, success_rate_af = \
        parse_results(train_scores_af, test_scores_af)
    mae_train_dc, mae_test_dc, success_rate_dc = \
        parse_results(train_scores_dc, test_scores_dc)
    benchmark_names = [benchmark.name for benchmark in suite]

    print_results("MAE (Train)", mae_train_af, mae_train_dc, benchmark_names)
    print_results("MAE (Test)", mae_test_af, mae_test_dc, benchmark_names)
    print_results("Success Rate", success_rate_af, success_rate_dc,
                  benchmark_names)


if __name__ == "__main__":
    MAX_GENERATIONS = 200
    NUM_REPEATS = 2
    run_benchmark_comparison()
