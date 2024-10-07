from bingo.symbolic_regression.symbolic_regressor import SymbolicRegressor
from bingo.symbolic_regression.srbench_interface import (
    model,
    get_population,
    get_best_solution,
    eval_kwargs,
)

"""
est: a sklearn-compatible regressor.
"""
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
    max_time=350,
    max_evals=int(1e19),
    evolutionary_algorithm="AgeFitnessEA",
    clo_threshold=1.0e-5,
)
