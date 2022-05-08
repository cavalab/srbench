import os
from datetime import datetime
from copy import deepcopy
import multiprocessing
from itertools import compress, chain

from sympy import lambdify, preorder_traversal
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import train_test_split
import numpy as np

from dso import DeepSymbolicOptimizer
from dso.config import load_config
from dso.program import Program
from dso.utils import is_pareto_efficient


def work(args):
    config, X, y, max_time = args
    est = UnifiedDeepSymbolicRegressor(config)
    est.fit(X, y, max_time=max_time)
    pf = est.pf
    # expr = est.program_.sympy_expr[0]
    # r = est.program_.r
    return pf


class ParallelizedUnifiedDeepSymbolicRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.base_config = load_config()

        try:
            self.n_cpus = os.environ['OMP_NUM_THREADS']
        except KeyError:
            self.n_cpus = 4

    def fit(self, X, y, max_time=None):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

        # y_train = np.sin(X_train[:, 0]) + 1.23*X_train[:, 1] + 4.56*X_train[:, 2]
        # X = np.random.random((100, 3))
        # y = np.sin(X[:, 0]) + 1.23*X[:, 1] + 4.56*X[:, 2]

        # Generate the configs
        configs = self.make_configs(X, y)
        args = tuple((config, X_train, y_train, max_time) for config in configs)

        # Farm out the work
        # NOTE: This assumes each worker will get exactly one job
        pool = multiprocessing.Pool(self.n_cpus)
        pfs = pool.map(work, args)
        pfs = list(chain.from_iterable(pfs))

        # Pool the Pareto fronts

        # Evaluate on the test data
        best_expr = None
        best_f = None
        best_mse = np.inf
        variables = ["x{}".format(i+1) for i in range(X.shape[1])]
        for expr in pfs:            
            f = lambdify(variables, expr)
            y_hat_test = f(*X_test.T)
            test_mse = np.mean(np.square(y_hat_test - y_test))
            print("LAMBDIFY OUTPUT", expr, "MSE_TEST:", test_mse, "COMPLEXITY:", len(list(preorder_traversal(expr))))
            if test_mse < best_mse:
                best_mse = test_mse
                best_expr = expr
                best_f = f

        # Choose the best expression
        self.expr = best_expr
        self.f = best_f

        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self, "f")
        return self.f(*X.T)

    def make_configs(self, X, y):
        """
        Generate different configs based on Xy data and n_cpus.
        """
        configs = []
        for i in range(self.n_cpus):
            config = deepcopy(self.base_config)
            config["experiment"]["seed"] = i # Always use a different seed

            # Base
            if i == 0:
                pass

            # # No poly
            # elif i == 1:
            #     config["task"]["function_set"].remove("poly")

            configs.append(config)
        return configs


class UnifiedDeepSymbolicRegressor(BaseEstimator, RegressorMixin):
    """
    Sklearn interface for unified deep symbolic regression (uDSR).
    Used for SRBench 2022 competition.
    """

    def __init__(self, config=None):
        self.config = load_config() if config is None else config

    def fit(self, X, y, max_time=None):

        # Competition guide: max time is based on X.shape[0]
        if max_time is None:
            max_time = 10 * 60 * 60 if X.shape[0] > 1000 else 1 * 60 * 60

        # Update the config
        self.config["task"]["dataset"] = (X, y)
        self.config["experiment"]["max_time"] = max_time

        # Create the model
        model = DeepSymbolicOptimizer(self.config)
        train_result = model.train()
        self.program_ = train_result["program"]
        self.pf = self.get_pareto_front()

        print("(pid={}) finished. Reward: {}. Model: {}".format(multiprocessing.current_process(), self.program_.r, self.program_.sympy_expr[0]))

        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self, "program_")
        return self.program_.execute(X)

    def get_pareto_front(self):
        all_programs = list(Program.cache.values())
        costs = np.array([(p.complexity, -p.r) for p in all_programs])
        pareto_efficient_mask = is_pareto_efficient(costs)  # List of bool
        pf = list(compress(all_programs, pareto_efficient_mask))

        # Compute sympy expressions
        pf = [p.sympy_expr[0] for p in pf]
        pf.sort(key=lambda expr: len(list(preorder_traversal(expr)))) # Sort by sympy-complexity
        return pf
