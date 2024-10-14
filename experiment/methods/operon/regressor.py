import copy
import numpy as np
import optuna
import os
import pandas as pd
import time

import pyoperon as Operon
from pyoperon.sklearn import SymbolicRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, RegressorMixin
from sympy import lambdify, parse_expr, Symbol


def compute_sigma(X, y):
    rf = RandomForestRegressor()
    rf.fit(X, y)
    return np.sqrt(mean_squared_error(y, rf.predict(X)))


def compute_mdl(tree, X, y):
    ds = Operon.Dataset(np.asfortranarray(np.column_stack((X, y))))
    problem = Operon.Problem(ds)
    problem.TrainingRange = Operon.Range(0, X.shape[0])
    target = max(ds.Variables, key=lambda x: x.Index)
    problem.Target = target
    problem.InputHashes = [
        v.Hash
        for v in sorted(ds.Variables, key=lambda x: x.Index)
        if v.Hash != target.Hash
    ]
    dtable = Operon.DispatchTable()
    mdl_eval = Operon.MinimumDescriptionLengthEvaluator(problem, dtable, "gauss")
    mdl_eval.Sigma = [compute_sigma(X, y)]
    rng = Operon.RomuTrio(1234)
    ind = Operon.Individual()
    ind.Genotype = tree
    return mdl_eval(rng, ind)[0]


class SympyExprModel(RegressorMixin):
    """Simple class encapsulating a symbolic expression model compiled into a Sympy lambda

    Attributes:
    ----------

    is_fitted: str
        This regressor is already fitted (the encapsulated expression is a trained model)


    Methods:
    -------

    __init__(expr_str, variable_names):
        Constructor which builds a lambda from the expression string using the specified variable names.

    """

    def __init__(self, expr_str, variable_names):
        self.model = expr_str
        self.variable_names = variable_names
        self.symbols = [Symbol(x) for x in variable_names]
        self.best_estimator = lambdify(self.symbols, parse_expr(expr_str))
        self.is_fitted = True

    def predict(self, X):
        values = X.values if isinstance(X, pd.DataFrame) else X
        return self.best_estimator(*values.T)


class Objective:
    def __init__(self, X, y, n_folds, seed, fixed_params={}, suggest_callback=None):
        self.X = X
        self.y = y
        self.n_folds = n_folds
        self.best_reg = None
        self.reg = None
        self.seed = seed
        self.params = fixed_params
        self.suggest_callback = suggest_callback

    def __call__(self, trial):
        hyper_params = (
            {} if self.suggest_callback is None else self.suggest_callback(trial)
        )
        params = copy.deepcopy(self.params)
        params.update(hyper_params)
        params.update({"random_state": self.seed})
        self.reg = SymbolicRegressor(**params)

        X_, y_ = np.asarray(self.X), np.asarray(self.y)

        if self.n_folds > 1:
            kf = KFold(n_splits=self.n_folds)
            score = 0

            for train, test in kf.split(X_, y_):
                X_train, y_train = X_[train, :], y_[train]
                X_test, y_test = X_[test, :], y_[test]
                self.reg.fit(X_train, y_train)
                best = min(
                    self.reg.pareto_front_,
                    key=lambda x: x["minimum_description_length"],
                )
                score += compute_mdl(best["tree"], X_test, y_test)

            return score / self.n_folds

        self.reg.fit(X_, y_)
        return min((x["minimum_description_length"] for x in self.reg.pareto_front_))

    def callback(self, study, trial):
        if self.best_reg is None or study.best_trial == trial:
            self.best_reg = self.reg


class OperonOptuna(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        name,
        n_folds=3,
        n_trials=100,
        max_time=3600,
        fixed_params={},
        suggest_callback=None,
        random_state=None,
    ):
        self.name = name
        self.n_folds = n_folds
        self.n_trials = n_trials
        self.max_time = max_time
        self.random_state = random_state
        self.fixed_params = fixed_params
        self.suggest_callback = suggest_callback
        self.best_estimator = None  # to hold the best estimator
        self.best_parameters = None  # to hold the best parameters

    def fit(self, X, y):
        t0 = time.time()
        sErr = compute_sigma(X, y)
        fixed_params = copy.deepcopy(self.fixed_params)
        fixed_params.update({"uncertainty": [sErr]})
        t1 = time.time()
        curr_time = np.ceil(t1 - t0)

        objective = Objective(
            X,
            y,
            self.n_folds,
            fixed_params=fixed_params,
            suggest_callback=self.suggest_callback,
            seed=self.random_state,
        )
        study = optuna.create_study(study_name=self.name, direction="minimize")
        study.optimize(
            objective,
            n_trials=self.n_trials,
            callbacks=[objective.callback],
            timeout=(self.max_time - curr_time),
        )
        self.best_estimator = objective.best_reg
        self.best_parameters = self.best_estimator.get_params()
        self.best_parameters["completed trials"] = len(study.trials)

        # save the best solution as a SympyExprModel
        best = self.best_estimator
        names = (
            X.columns.tolist()
            if isinstance(X, pd.DataFrame)
            else list(best.variables_.values())
        )
        model_string = lambda x: best.get_model_string(x, 6, names).replace("^", "**")

        self.best_solution = SympyExprModel(model_string(best.model_), names)
        self.model = self.best_solution.model
        self.population = [
            SympyExprModel(model_string(x["tree"]), names)
            for x in best.pareto_front_[:100]
        ]

    def predict(self, X):
        return self.best_estimator.predict(X)


num_threads = (
    int(os.environ["OMP_NUM_THREADS"]) if "OMP_NUM_THREADS" in os.environ else 1
)

default_params = {
    "offspring_generator": "basic",
    "initialization_method": "btc",
    "n_threads": num_threads,
    "objectives": ["r2", "length"],
    "epsilon": 1e-5,
    "optimizer_iterations": 1,
    "random_state": None,
    "reinserter": "keep-best",
    "max_evaluations": int(1e6),
    "female_selector": "tournament",
    "male_selector": "tournament",
    "brood_size": 5,
    "population_size": 1000,
    "pool_size": None,
    "max_time": 300,
    "model_selection_criterion": "minimum_description_length",
}


def suggest_params(trial):
    return {
        "allowed_symbols": trial.suggest_categorical(
            "allowed_symbols",
            [
                "add,sub,mul,div,constant,variable",
                "add,sub,mul,div,sin,cos,constant,variable",
                "add,sub,mul,div,exp,log,constant,variable",
                "add,sub,mul,div,sin,cos,exp,log,sqrt,pow,constant,variable",
            ],
        ),
        "max_length": trial.suggest_int("max_length", low=10, high=50, step=10),
    }


def model(est, X=None):
    return est.model


def get_best_solution(est):
    return [est] if isinstance(est, SympyExprModel) else est.best_solution


def get_population(est):
    return [est] if isinstance(est, SympyExprModel) else est.population


est = OperonOptuna(
    "operon-optuna",
    n_folds=1,
    n_trials=100,
    fixed_params=default_params,
    suggest_callback=suggest_params,
)

################################################################################
# Optional Settings
################################################################################

"""
eval_kwargs: a dictionary of variables passed to the evaluate_model()
    function.
    Allows one to configure aspects of the training process.

Options
-------
    test_params: dict, default = None
        Used primarily to shorten run-times during testing.
        for running the tests. called as
            est = est.set_params(**test_params)
    max_train_samples:int, default = 0
        if training size is larger than this, sample it.
        if 0, use all training samples for fit.
    scale_x: bool, default = True
        Normalize the input data prior to fit.
    scale_y: bool, default = True
        Normalize the input label prior to fit.
    pre_train: function, default = None
        Adjust settings based on training data. Called prior to est.fit.
        The function signature should be (est, X, y).
            est: sklearn regressor; the fitted model.
            X: pd.DataFrame; the training data.
            y: training labels.
"""


def pre_train_fn(est, X, y):
    pass


# pass the function to eval_kwargs
eval_kwargs = {"pre_train": pre_train_fn, "test_params": {"max_time": 60}}
