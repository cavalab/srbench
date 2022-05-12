import os
from math import factorial
from time import time
from copy import deepcopy
import multiprocessing
from itertools import chain

from sympy import lambdify, preorder_traversal
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import train_test_split
import numpy as np
import threadpoolctl

from dso import DeepSymbolicOptimizer
from dso.config import load_config


def work(args):
    # Create the regressor
    config, X, y, max_time = args
    est = UnifiedDeepSymbolicRegressor(config)

    # Fit the regressor
    try:
        est.fit(X, y, max_time=max_time)        
    except Exception as e:
        print(multiprocessing.current_process())
        print("WORKER ERROR DURING CALL TO FIT:", e)
        print("Worker config:", config)

    # Get the Pareto front
    try:
        pf = est.get_pf()
    except Exception as e:
        print(multiprocessing.current_process())
        print("WORKER HAS NO PARETO FRONT:", e)
        print("Worker config:", config)
        pf = []

    print(multiprocessing.current_process(), "Returning Pareto front of length {}.".format(len(pf)))

    return pf


class ParallelizedUnifiedDeepSymbolicRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.base_config = load_config()

        try:
            self.n_cpus = int(os.environ['OMP_NUM_THREADS'])
        except KeyError:
            self.n_cpus = 8

    def fit(self, X, y, max_time=None):

        # Triage: Train-test split
        if X.shape[0] >= 100:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        else:
            print("TRIAGE: Too few data points to perform train-test split.")
            X_train = X_test = X
            y_train = y_test = y

        # Triage: Polynomial terms
        nCr = lambda n, r : factorial(n) / factorial(r) / factorial(n - r)
        n_var = X.shape[1]
        degree = self.base_config["task"]["poly_optimizer_params"]["degree"]
        while nCr(n_var + degree - 1, degree) > 1000:
            degree -= 1
            print("TRIAGE: Lowering degree to {} to yield fewer terms.".format(degree))
        self.base_config["task"]["poly_optimizer_params"]["degree"] = degree

        # Generate the configs
        configs = self.make_configs(X, y)
        args = tuple((config, X_train, y_train, max_time) for config in configs)

        # Farm out the work
        # NOTE: This assumes each worker will get exactly one job
        pool = multiprocessing.Pool(self.n_cpus)
        pfs = pool.map(work, args)

        start = time()

        # Pool the Pareto fronts
        pfs = list(chain(*pfs))

        # Evaluate on the test data
        best_expr = None
        best_f = None
        best_nmse = np.inf
        best_complexity = np.inf
        variables = ["x{}".format(i + 1) for i in range(X.shape[1])]
        var_y_test = np.var(y_test)
        for expr in pfs:            
            f = lambdify(variables, expr)
            y_hat_test = f(*X_test.T)
            complexity = len(list(preorder_traversal(expr)))
            test_nmse = np.mean(np.square(y_hat_test - y_test)) / var_y_test
            print("COMPLEXITY:", complexity, "NMSE_TEST:", test_nmse, "LAMBDIFY OUTPUT", expr)
            if test_nmse < 1e-16 and complexity < best_complexity:
                best_nmse = 0
                best_complexity = complexity
                best_expr = expr
                best_f = f
            elif test_nmse < best_nmse:
                best_nmse = test_nmse
                best_complexity = complexity
                best_expr = expr
                best_f = f

        print("DEBUG: Evaluation on Pareto front took {} seconds.".format(time() - start))

        # Choose the best expression
        self.expr = best_expr
        self.f = best_f

        print("DEBUG: Best model:", self.expr, "with complexity", best_complexity)

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
            config["experiment"]["seed"] = np.random.randint(1e9) # Always use a different seed

            # Standard config
            if i < 3:
                pass

            # No const/poly, LR=0
            elif i == 3:
                config["task"]["function_set"].remove("const")
                config["task"]["function_set"].remove("poly")
                config["controller"]["learning_rate"] = 0.0

            # No const/poly
            elif i == 4:
                config["task"]["function_set"].remove("const")
                config["task"]["function_set"].remove("poly")

            # No const
            elif i == 5:
                config["task"]["function_set"].remove("const")

            # No sin/cos?
            elif i == 6:
                config["task"]["function_set"].remove("sin")
                config["task"]["function_set"].remove("cos")

            # No exp/log
            elif i == 7:
                config["task"]["function_set"].remove("exp")
                config["task"]["function_set"].remove("log")

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
        with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
            return self._fit(X, y, max_time)

    def _fit(self, X, y, max_time=None):

        # Competition guide: max time is based on X.shape[0]
        if max_time is None:
            max_time = 10 * 60 * 60 if X.shape[0] > 1000 else 1 * 60 * 60

        # Update the config
        self.config["task"]["dataset"] = (X, y)
        self.config["experiment"]["max_time"] = max_time

        # Create the model
        self.model = DeepSymbolicOptimizer(self.config)
        self.model.train()

        self.is_fitted_ = True
        return self

    def get_pf(self):
        return self.model.pf.get_sympy()
