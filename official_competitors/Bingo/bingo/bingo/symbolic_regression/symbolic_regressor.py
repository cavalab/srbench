import numpy as np
import os
import signal

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import GridSearchCV

from bingo.evaluation.evaluation import Evaluation
from bingo.evolutionary_algorithms.age_fitness import AgeFitnessEA
from bingo.evolutionary_algorithms.deterministic_crowding import DeterministicCrowdingEA
from bingo.evolutionary_optimizers.fitness_predictor_island import FitnessPredictorIsland
from bingo.evolutionary_optimizers.island import Island
from bingo.evolutionary_optimizers.parallel_archipelago import ParallelArchipelago
from bingo.evolutionary_optimizers.serial_archipelago import SerialArchipelago
from bingo.local_optimizers.continuous_local_opt import ContinuousLocalOptimization
from bingo.stats.hall_of_fame import HallOfFame
from bingo.symbolic_regression.agraph.component_generator import ComponentGenerator
from bingo.symbolic_regression.agraph.crossover import AGraphCrossover
from bingo.symbolic_regression.agraph.mutation import AGraphMutation
from bingo.symbolic_regression import AGraphGenerator #, ExplicitRegression, ExplicitTrainingData
from bingo.symbolic_regression.explicit_regression import ExplicitRegression, ExplicitTrainingData # this forces use of python fit funcs

from bingo.evolutionary_optimizers.fitness_predictor_island import FitnessPredictorIsland
from bingo.evolutionary_optimizers.island import Island


class SymbolicRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, population_size=500, stack_size=32,
                 operators=None, use_simplification=False,
                 crossover_prob=0.4, mutation_prob=0.4,
                 metric="mse", parallel=False, clo_alg="lm",
                 generations=int(1e19), fitness_threshold=1.0e-16,
                 max_time=1800, max_evals=int(1e19), evolutionary_algorithm=AgeFitnessEA,
                 clo_threshold=1.0e-8, scale_max_evals=False, random_state=None):
        self.population_size = population_size
        self.stack_size = stack_size

        if operators is None:
            operators = {"+", "-", "*", "/"}
        self.operators = operators

        self.use_simplification = use_simplification

        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob

        self.metric = metric

        self.parallel = parallel

        self.clo_alg = clo_alg

        self.generations = generations
        self.fitness_threshold = fitness_threshold
        self.max_time = max_time
        self.max_evals = max_evals
        self.scale_max_evals = scale_max_evals

        self.evolutionary_algorithm = evolutionary_algorithm

        self.clo_threshold = clo_threshold

        self.best_ind = None

        self.random_state = random_state

        if random_state is not None:
            np.random.seed(random_state)
            random.seed(random_state)

    def set_params(self, **params):
        # TODO not clean
        new_params = self.get_params()
        new_params.update(params)
        super().set_params(**new_params)
        self.__init__(**new_params)
        return self

    def set_max_time(self, new_max_time):
        self.max_time = new_max_time

    def _get_archipelago(self, X, y, n_processes):
        self.component_generator = ComponentGenerator(X.shape[1])
        for operator in self.operators:
            self.component_generator.add_operator(operator)

        self.crossover = AGraphCrossover()
        self.mutation = AGraphMutation(self.component_generator)

        self.generator = AGraphGenerator(self.stack_size, self.component_generator,
                                         use_simplification=self.use_simplification,
                                         use_python=True # only using c++ backend
                                         )

        local_opt_fitness = self._get_clo(X, y, self.clo_threshold)
        evaluator = Evaluation(local_opt_fitness, n_processes)

        if self.evolutionary_algorithm == AgeFitnessEA:
            ea = self.evolutionary_algorithm(evaluator, self.generator, self.crossover,
                                             self.mutation, self.crossover_prob, self.mutation_prob,
                                             self.population_size)

        else:  # DeterministicCrowdingEA
            ea = self.evolutionary_algorithm(evaluator, self.crossover, self.mutation,
                                             self.crossover_prob, self.mutation_prob)

        # TODO pareto front based on complexity?
        hof = HallOfFame(5)

        island = self.make_island(len(X), ea, hof)
        self._force_diversity_in_island(island)

        if self.parallel:
            return ParallelArchipelago(island, hall_of_fame=hof)
        else:
            return island

    def make_island(self, dset_size, ea, hof):
        if dset_size < 1200:
            return Island(ea, self.generator, self.population_size, hall_of_fame=hof)
        return FitnessPredictorIsland(ea, self.generator, self.population_size, hall_of_fame=hof,
                                      predictor_size_ratio=800/dset_size)


    def _get_clo(self, X, y, tol):
        training_data = ExplicitTrainingData(X, y)
        fitness = ExplicitRegression(training_data=training_data, metric=self.metric)
        local_opt_fitness = ContinuousLocalOptimization(fitness, algorithm=self.clo_alg, tol=tol)
        return local_opt_fitness


    def _force_diversity_in_island(self, island):
        diverse_pop = []
        pop_strings = set()

        i = 0
        while len(diverse_pop) < self.population_size:
            i+=1
            ind = self.generator()
            ind_str = str(ind)
            if ind_str not in pop_strings or i > 15 * self.population_size:
                pop_strings.add(ind_str)
                diverse_pop.append(ind)
        print(f" Generating a diverse population took {i} iterations.")
        island.population = diverse_pop

    def get_best_individual(self):
        if self.best_ind is None:
            print("Best individual is None, setting to X_0")
            from bingo.symbolic_regression import AGraph
            self.best_ind = AGraph()
            self.best_ind.command_array = np.array([[0, 0, 0]], dtype=int)  # X_0
        return self.best_ind

    def fit(self, X, y, sample_weight=None):
        if sample_weight is not None:
            print("sample weight not None, TODO")
            raise NotImplementedError

        try:
            n_cpus = os.environ['OMP_NUM_THREADS']
        except KeyError:
            n_cpus = 1
        print(f"using {n_cpus} processes")
        self.archipelago = self._get_archipelago(X, y, n_cpus)
        print(f"archipelago: {type(self.archipelago)}")

        max_eval_scaling = 1
        if isinstance(self.archipelago, FitnessPredictorIsland):
            print("n_points per predictor:", self.archipelago._predictor_size)
            if self.scale_max_evals:
                max_eval_scaling = len(X) / self.archipelago._predictor_size / 1.1
        # print("max evals:", self.max_evals * max_eval_scaling)

        opt_result = self.archipelago.evolve_until_convergence(
            max_generations=self.generations,
            fitness_threshold=self.fitness_threshold,
            max_time=self.max_time,
            max_fitness_evaluations=self.max_evals * max_eval_scaling,
            convergence_check_frequency=10
        )

        if len(self.archipelago.hall_of_fame) == 0:  # most likely found sol in 0 gens
            self.best_ind = self.archipelago.get_best_individual()
        else:
            self.best_ind = self.archipelago.hall_of_fame[0]
        print(f"done with opt, best_ind: {self.best_ind}, fitness: {self.best_ind.fitness}")

        # rerun CLO on best_ind with tighter tol
        fit_func = self._get_clo(X, y, tol=1e-6)
        best_fitness = fit_func(self.best_ind)
        best_constants = tuple(self.best_ind.constants)
        for _ in range(5):
            self.best_ind._needs_opt = True
            fitness = fit_func(self.best_ind)
            if fitness < best_fitness:
                best_fitness = fitness
                best_constants = tuple(self.best_ind.constants)
        self.best_ind.fitness = best_fitness
        self.best_ind.set_local_optimization_params(best_constants)
        print(f"reran CLO, best_ind: {self.best_ind}, fitness: {self.best_ind.fitness}")

        # print("------------------hall of fame------------------", self.archipelago.hall_of_fame, sep="\n")
        # print("\nbest individual:", self.best_ind)

    def predict(self, X):
        output = self.best_ind.evaluate_equation_at(X)

        # convert nan to 0, inf to large number, and -inf to small number
        return np.nan_to_num(output, posinf=1e100, neginf=-1e100)


class TimeOutException(Exception):
    pass


def alarm_handler(signum, frame):
    print("raising TimeOutException")
    raise TimeOutException


class CrossValRegressor(GridSearchCV):
    def get_best_individual(self):
        return self.best_estimator_.get_best_individual()

    def set_max_time(self, new_max_time):
        self.estimator.set_params(**{"max_time": new_max_time})

    def fit(self, X, y=None, *, groups=None, **fit_params):
        signal.signal(signal.SIGALRM, alarm_handler)
        if len(X) <= 1000:  # probably not good to hardcode these values
            signal.alarm(60 * 60 - 5)  # 1 hour with 5 seconds of slack
            # signal.alarm(5)  # 1 hour with 5 seconds of slack
        else:
            signal.alarm(10 * 60 * 60 - 5)  # 10 hours with 5 seconds of slack
        try:
            super().fit(X, y, groups=groups, **fit_params)
        except TimeOutException as e:
            print("Got a TimeOutException")

            try:
                self.best_estimator_
            except AttributeError:  # no self.best_estimator
                # if we didn't get to the end of cv,
                # set best_estimator_ to the estimator passed in with the first
                # set of hyperparams
                print("Setting best estimator to first set of hyperparams")
                self.best_estimator_ = clone(self.estimator.set_params(**self.param_grid[0]))

            print(self.best_estimator_)
            return self


if __name__ == '__main__':
    # SRSerialArchipelagoExample with SymbolicRegressor
    import random
    from sklearn.model_selection import KFold
    # random.seed(7)
    # np.random.seed(7)
    x = np.linspace(-10, 10, 1000).reshape([-1, 1])
    y = x**2 + 3.5*x**3

    regr = SymbolicRegressor(population_size=100, stack_size=10,
                             operators=["+", "-", "*"],
                             use_simplification=True,
                             crossover_prob=0.4, mutation_prob=0.4, metric="mae",
                             parallel=False, clo_alg="lm", generations=500, fitness_threshold=1.0e-4,
                             evolutionary_algorithm=AgeFitnessEA, clo_threshold=1.0e-4, random_state=7)

    hyper_params = [
        {"population_size": [100], "stack_size": [24]},
        {"population_size": [500], "stack_size": [24]},
        {"population_size": [2500], "stack_size": [32]}
    ]

    cv = KFold(n_splits=3, shuffle=True)

    cv_regr = CrossValRegressor(regr, cv=cv, param_grid=hyper_params,
                                verbose=3, n_jobs=1, scoring="r2", error_score="raise")
    cv_regr.set_max_time(1)
    print(cv_regr.get_params())
    print(cv_regr)

    cv_regr.fit(x, y)
    print(cv_regr.get_best_individual())

    # TODO MPI.COMM_WORLD.bcast in parallel?
    # TODO rank in MPI
