from __future__ import annotations

from typing import Any

from bingo.symbolic_regression.srbench_interface import (
    eval_kwargs as base_eval_kwargs,
)
from bingo.symbolic_regression.srbench_interface import (
    get_best_solution as get_best_solution,
)
from bingo.symbolic_regression.srbench_interface import (
    get_population as get_population,
)
from bingo.symbolic_regression.srbench_interface import (
    model as model,
)
from bingo.symbolic_regression.symbolic_regressor import SymbolicRegressor

eval_kwargs = dict(base_eval_kwargs)
eval_kwargs["use_dataframe"] = False

# sklearn's interface should not modify constructor parameters. Bingo modifies
# max_time, so here I'm wrapping it into an sklearn-compatible estimator
# so gridsearch will work.

# class BingoSymbolicRegressor(SymbolicRegressor, RegressorMixin):
#     def __init__(
#         self,
#         *,
#         population_size=500,
#         stack_size=32,
#         operators=None,
#         use_simplification=False,
#         crossover_prob=0.4,
#         mutation_prob=0.4,
#         metric="mse",
#         clo_alg="lm",
#         generations=int(1e19),
#         fitness_threshold=1.0e-16,
#         max_time=1800,
#         max_evals=int(1e19),
#         evolutionary_algorithm=None,
#         clo_threshold=1.0e-5,
#         scale_max_evals=False,
#         random_state=None,
#     ):
#         self.population_size = population_size
#         self.stack_size = stack_size

#         if operators is None:
#             operators = DEFAULT_OPERATORS
#         self.operators = operators

#         self.use_simplification = use_simplification

#         self.crossover_prob = crossover_prob
#         self.mutation_prob = mutation_prob

#         self.metric = metric

#         self.clo_alg = clo_alg

#         self.generations = generations
#         self.fitness_threshold = fitness_threshold
#         self.max_time = max_time # * TIME_REDUCTION_FACTOR
#         self.max_evals = max_evals
#         self.scale_max_evals = scale_max_evals

#         self.evolutionary_algorithm = evolutionary_algorithm

#         self.clo_threshold = clo_threshold

#         self.random_state = random_state


#     def fit(self, X, y, sample_weight=None):
#         self.est_ = SymbolicRegressor(
#             population_size=self.population_size,
#             stack_size=self.stack_size,
#             operators=self.operators,
#             use_simplification=self.use_simplification,
#             crossover_prob=self.crossover_prob,
#             mutation_prob=self.mutation_prob,
#             metric=self.metric,
#             # parallel=False,
#             clo_alg=self.clo_alg,
#             max_time=self.max_time,
#             max_evals=self.max_evals,
#             evolutionary_algorithm=self.evolutionary_algorithm,
#             clo_threshold=self.clo_threshold,
#         )

#         self.est_.fit(X, y, sample_weight)

#         self.best_pop_ = self.est_._find_best_population(X, y)
#         self.best_ind_ = min(self.est_.best_pop_, key=lambda x: x.fitness)

#         return self

#     def get_params(self, deep=True):
#         out = dict()
#         for (key, value) in self.__dict__.items():
#             if not key.endswith('_'):
#                 if deep and hasattr(value, "get_params") and not isinstance(value, type):
#                     deep_items = value.get_params().items()
#                     out.update((key + "__" + k, val) for k, val in deep_items)
#                 out[key] = value
#         return out

#     def get_best_individual(self):
#         if self.best_ind_ is None:
#             raise ValueError("Best individual not set. Make sure fit() was called.")
#         return self.best_ind_

#     def get_best_population(self):
#         if self.best_pop_ is None:
#             raise ValueError("Best population not set. Make sure fit() was called.")
#         return self.best_pop_

#     def get_pareto_front(self):
#         if self.best_pop_ is None:
#             raise ValueError("Pareto front not set. Make sure fit() was called.")
#         hof = ParetoFront(
#             secondary_key=lambda equ: equ.complexity,
#             similarity_function=lambda x, y: x.fitness == y.fitness
#             and x.complexity == y.complexity,
#         )
#         hof.update(self.best_pop_)
#         return list(hof)

#     def predict(self, X):
#         best_ind = self.get_best_individual()
#         return best_ind.predict(X)

# """
# est: a sklearn-compatible regressor.
# """
# est = BingoSymbolicRegressor(
#     population_size=500,
#     stack_size=24,
#     operators=["+", "-", "*", "/", "sin", "cos", "exp", "log", "sqrt"],
#     use_simplification=True,
#     crossover_prob=0.3,
#     mutation_prob=0.45,
#     metric="mse",
#     # parallel=False,
#     clo_alg="lm",
#     max_time=60*60,
#     max_evals=int(1e19),
#     evolutionary_algorithm="AgeFitnessEA",
#     clo_threshold=1.0e-5,
# )

population_size = [100, 500, 1000]
operators = [
    ["+", "-", "*", "/", "sin", "cos", "exp", "log", "sqrt"],
    ["+", "-", "*", "/", "log", "sqrt"],
]

hyper_params: list[dict[str, Any]] = []
# for p in population_size:
#     for ops in operators:
#         hyper_params.append({
#             'population_size':[p],
#             'operators':[ops],
#         })

est = SymbolicRegressor(
    population_size=500,
    stack_size=24,
    operators=["+", "-", "*", "/", "sin", "cos", "exp", "log", "sqrt"],
    use_simplification=True,
    crossover_prob=0.3,
    mutation_prob=0.45,
    metric="mse",
    # parallel=False,
    clo_alg="lm",
    max_time=3500,
    max_evals=int(1e19),
    evolutionary_algorithm="AgeFitnessEA",
    clo_threshold=1.0e-5,
)
