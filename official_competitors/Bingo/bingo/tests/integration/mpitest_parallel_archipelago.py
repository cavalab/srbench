# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import sys
import os
import numpy as np
import inspect
import dill
from mpi4py import MPI
from unittest.mock import Mock
from bingo.chromosomes.multiple_values import SinglePointCrossover, \
                                              SinglePointMutation, \
                                              MultipleValueChromosomeGenerator
from bingo.evolutionary_optimizers.island import Island
from bingo.evolutionary_algorithms.mu_plus_lambda import MuPlusLambda
from bingo.selection.tournament import Tournament
from bingo.evaluation.evaluation import Evaluation
# from bingo.evaluation.fitness_function import FitnessFunction
from bingo.evolutionary_optimizers.parallel_archipelago \
    import ParallelArchipelago, load_parallel_archipelago_from_file

POP_SIZE = 5
SELECTION_SIZE = 10
VALUE_LIST_SIZE = 10
OFFSPRING_SIZE = 20
ERROR_TOL = 10e-6

COMM = MPI.COMM_WORLD
COMM_RANK = COMM.Get_rank()
COMM_SIZE = COMM.Get_size()


class MultipleValueFitnessFunction():
    def __init__(self, training_data=None):
        self.eval_count = 0
        self.training_data = training_data

    def __call__(self, individual):
        fitness = np.count_nonzero(individual.values)
        self.eval_count += 1
        return fitness


class NumberGenerator:
    def __init__(self, num):
        self.num = num

    def __call__(self):
        return self.num


def evol_alg():
    crossover = SinglePointCrossover()
    mutation = SinglePointMutation(NumberGenerator(-1))
    selection = Tournament(SELECTION_SIZE)
    fitness = MultipleValueFitnessFunction()
    evaluator = Evaluation(fitness)
    return MuPlusLambda(evaluator, selection, crossover, mutation,
                        0.2, 0.4, OFFSPRING_SIZE)


def num_island(num, pop_size=POP_SIZE):
    generator = MultipleValueChromosomeGenerator(NumberGenerator(num),
                                                 VALUE_LIST_SIZE)
    return Island(evol_alg(), generator, pop_size)


def perfect_individual():
    generator = MultipleValueChromosomeGenerator(NumberGenerator(0),
                                                 VALUE_LIST_SIZE)
    return generator()


def test_best_individual_returned():
    island = num_island(COMM_RANK + 1)
    if COMM_RANK == 0:
        island.population += [perfect_individual()]
    archipelago = ParallelArchipelago(island)
    return mpi_assert_equal(archipelago.get_best_individual().fitness, 0)


def test_best_fitness_returned():
    island = num_island(COMM_RANK + 1)
    if COMM_RANK == 0:
        island.population += [perfect_individual()]
    archipelago = ParallelArchipelago(island)
    return mpi_assert_equal(archipelago.get_best_fitness(), 0)


def test_potential_hof_members():
    island_a = Mock(hall_of_fame=[COMM_RANK, COMM_RANK])
    archipelago = ParallelArchipelago(num_island(1))
    archipelago.island = island_a
    actual_members = archipelago._get_potential_hof_members()
    expected_memebers = [i for i in range(COMM_SIZE) for _ in range(2)]
    return mpi_assert_equal(actual_members, expected_memebers)


def test_island_migration_doesnt_chane_pop_size():
    island = num_island(COMM_RANK, pop_size=COMM_RANK + 2)
    archipelago = ParallelArchipelago(island)
    archipelago._coordinate_migration_between_islands()
    expected_pop = (COMM_SIZE * (COMM_SIZE - 1)) // 2 + (2 * COMM_SIZE)
    total_pop_after = COMM.allreduce(len(archipelago.island.population),
                                     MPI.SUM)
    return mpi_assert_equal(total_pop_after, expected_pop)


def test_island_migration():
    island = num_island(COMM_RANK)
    archipelago = ParallelArchipelago(island)
    archipelago._coordinate_migration_between_islands()

    native_individual_values = [COMM_RANK]*VALUE_LIST_SIZE
    non_native_indv_found = False
    for individual in archipelago.island.population:
        if individual.values != native_individual_values:
            non_native_indv_found = True
            break
    has_unpaired_island = COMM_SIZE % 2 == 1
    if has_unpaired_island:
        return mpi_assert_exactly_n_false(non_native_indv_found, 1)
    return mpi_assert_true(non_native_indv_found)


def test_blocking_fitness_eval_count():
    steps = 1
    island = num_island(COMM_RANK)
    archipelago = ParallelArchipelago(island, non_blocking=False)
    archipelago.evolve(steps)
    expected_evaluations = COMM_SIZE * (POP_SIZE + steps * OFFSPRING_SIZE)
    actual_evaluations = archipelago.get_fitness_evaluation_count()
    return mpi_assert_equal(actual_evaluations, expected_evaluations)


def test_non_blocking_evolution():
    steps = 200
    island = num_island(COMM_RANK)
    archipelago = ParallelArchipelago(island, sync_frequency=10,
                                       non_blocking=True)
    archipelago.evolve(steps)
    island_age = archipelago.island.generational_age
    archipelago_age = archipelago.generational_age
    return mpi_assert_mean_near(island_age, archipelago_age, rel=0.1)


def test_convergence():
    island = num_island(COMM_RANK)
    archipelago = ParallelArchipelago(island, sync_frequency=10,
                                       non_blocking=True)
    result = archipelago.evolve_until_convergence(max_generations=100,
                                                  fitness_threshold=0,
                                                  convergence_check_frequency=25)
    return mpi_assert_true(result.success)


def test_dump_then_load_equal_procs():
    island = num_island(COMM_RANK)
    archipelago = ParallelArchipelago(island, sync_frequency=10,
                                       non_blocking=True)
    file_name = "testing_pa_dump_and_load_eq.pkl"
    archipelago.dump_to_file(file_name)
    archipelago = \
        load_parallel_archipelago_from_file(file_name)
    if COMM_RANK == 0:
        os.remove(file_name)

    origin_proc = archipelago.island.population[0].values[0]
    return mpi_assert_equal(origin_proc, COMM_RANK)


def test_dump_then_load_more_procs():
    island = num_island(COMM_RANK)
    archipelago = ParallelArchipelago(island, sync_frequency=10,
                                       non_blocking=True)
    file_name = "testing_pa_dump_and_load_gt.pkl"
    archipelago.dump_to_file(file_name)
    _remove_proc_from_pickle(file_name)
    archipelago = \
        load_parallel_archipelago_from_file(file_name)
    if COMM_RANK == 0:
        os.remove(file_name)

    origin_proc = archipelago.island.population[0].values[0]
    expected_origin = COMM_RANK + 1
    if COMM_RANK == COMM_SIZE - 1:
        expected_origin = 1
    return mpi_assert_equal(origin_proc, expected_origin)


def _remove_proc_from_pickle(file_name):
    if COMM_RANK == 0:
        with open(file_name, "rb") as pkl_file:
            par_arch_list = dill.load(pkl_file)
        par_arch_list.pop(0)
        with open(file_name, "wb") as pkl_file:
            dill.dump(par_arch_list, pkl_file, protocol=dill.HIGHEST_PROTOCOL)


def test_dump_then_load_less_procs():
    island = num_island(COMM_RANK)
    archipelago = ParallelArchipelago(island, sync_frequency=10,
                                       non_blocking=True)
    file_name = "testing_pa_dump_and_load_lt.pkl"
    archipelago.dump_to_file(file_name)
    _add_proc_to_pickle(file_name)
    archipelago = \
        load_parallel_archipelago_from_file(file_name)
    if COMM_RANK == 0:
        os.remove(file_name)

    origin_proc = archipelago.island.population[0].values[0]
    expected_origin = (COMM_RANK + 1) % COMM_SIZE
    return mpi_assert_equal(origin_proc, expected_origin)


def _add_proc_to_pickle(file_name):
    if COMM_RANK == 0:
        with open(file_name, "rb") as pkl_file:
            par_arch_list = dill.load(pkl_file)
        par_arch_list += par_arch_list[:2]
        par_arch_list.pop(0)
        with open(file_name, "wb") as pkl_file:
            dill.dump(par_arch_list, pkl_file, protocol=dill.HIGHEST_PROTOCOL)


def test_stale_checkpoint_removal():
    island = num_island(COMM_RANK)
    archipelago = ParallelArchipelago(island, non_blocking=False)
    archipelago.evolve_until_convergence(3, -1.0, num_checkpoints=1,
                                         checkpoint_base_name="stale_check")
    COMM.barrier()
    correct_files = [not os.path.isfile("stale_check_0.pkl"),
                     not os.path.isfile("stale_check_1.pkl"),
                     not os.path.isfile("stale_check_2.pkl"),
                     os.path.isfile("stale_check_3.pkl")]
    COMM.barrier()
    if COMM_RANK == 0:
        os.remove("stale_check_3.pkl")
    return mpi_assert_true(all(correct_files))

# ============================================================================


def mpi_assert_equal(actual, expected):
    equal = actual == expected
    if not equal:
        message = "\tproc {}:  {} != {}\n".format(COMM_RANK, actual, expected)
    else:
        message = ""
    all_equals = COMM.allgather(equal)
    all_messages = COMM.allreduce(message, op=MPI.SUM)
    return all(all_equals), all_messages


def mpi_assert_true(value):
    if not value:
        message = "\tproc {}: False, expected True\n".format(COMM_RANK)
    else:
        message = ""
    all_values = COMM.allgather(value)
    all_messages = COMM.allreduce(message, op=MPI.SUM)
    return all(all_values), all_messages


def mpi_assert_exactly_n_false(value, n):
    all_values = COMM.allgather(value)
    if sum(all_values) == len(all_values) - n:
        return True, ""

    message = "\tproc {}: {}\n".format(COMM_RANK, value)
    all_messages = COMM.allreduce(message, op=MPI.SUM)
    all_messages = "\tExpected exactly " + str(n) + " False\n" + all_messages
    return False, all_messages


def mpi_assert_mean_near(value, expected_mean, rel=1e-6, abs=None):
    actual_mean = COMM.allreduce(value, op=MPI.SUM)
    actual_mean /= COMM_SIZE
    allowable_error = rel * expected_mean
    if abs is not None:
        allowable_error = max(allowable_error, abs)

    if -allowable_error <= actual_mean - expected_mean <= allowable_error:
        return True, ""

    message = "\tproc {}:  {}\n".format(COMM_RANK, value)
    all_messages = COMM.allreduce(message, op=MPI.SUM)
    all_messages += "\tMean {} != {} +- {}".format(actual_mean, expected_mean,
                                                   allowable_error)
    return False, all_messages


def run_t(test_name, test_func):
    if COMM_RANK == 0:
        print(test_name, end=" ")
    COMM.barrier()
    success, message = test_func()
    COMM.barrier()
    if success:
        if COMM_RANK == 0:
            print(".")
    else:
        if COMM_RANK == 0:
            print("F")
            print(message, end=" ")
    return success


def driver():
    results = []
    tests = [(name, func)
             for name, func in inspect.getmembers(sys.modules[__name__],
                                                  inspect.isfunction)
             if "test" in name]
    if COMM_RANK == 0:
        print("========== collected", len(tests), "items ==========")

    for name, func in tests:
        results.append(run_t(name, func))

    num_success = sum(results)
    num_failures = len(results) - num_success
    if COMM_RANK == 0:
        print("==========", end="  ")
        if num_failures > 0:
            print(num_failures, "failed,", end=" ")
        print(num_success, "passed ==========")

    if num_failures > 0:
        exit(-1)


if __name__ == "__main__":
    driver()