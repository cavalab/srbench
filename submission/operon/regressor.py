from operon.sklearn import SymbolicRegressor
import optuna
import pandas as pd

import os
num_threads = os.environ['OMP_NUM_THREADS'] if 'OMP_NUM_THREADS' in os.environ else 1

default_params = {
        'offspring_generator': 'basic',
        'initialization_method': 'btc',
        'n_threads': num_threads,
        'objectives':  ['r2', 'length'],
        'epsilon':  1e-4,
        'random_state': None,
        'reinserter': 'keep-best',
        'max_evaluations': int(1e6),
        'tournament_size': 3,
        'pool_size': None,
        'time_limit': 2700 # 45 min
        }


# define parameter distributions
param_distributions = {
        'local_iterations' : optuna.distributions.IntUniformDistribution(0, 10, 1),
        'allowed_symbols' : optuna.distributions.CategoricalDistribution(['add,sub,mul,div,constant,variable', 'add,sub,mul,div,sin,cos,exp,logabs,sqrtabs,tanh,constant,variable']),
        'population_size' : optuna.distributions.IntUniformDistribution(100, 1000, 100),
        'max_length' : optuna.distributions.IntUniformDistribution(10, 50, 10),
        'symbolic_mode' : optuna.distributions.CategoricalDistribution([False, True]),
        }

# want to tune your estimator? wrap it in a sklearn CV class.
reg = SymbolicRegressor(**default_params)
est = optuna.integration.OptunaSearchCV(reg, param_distributions, cv=5, refit=True, n_trials=50, timeout=3000)

def model(est, X=None):
    names = X.columns.tolist() if isinstance(X, pd.DataFrame) else None
    return est.best_estimator_.get_model_string(precision=4, names=names).replace('^', '**')

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
    #"""set max_time in seconds based on length of X."""
    timeout = 3000 if len(X) <= 1000 else 30000
    est.set_params(timeout=timeout, estimator__time_limit=timeout-300)


# pass the function to eval_kwargs
eval_kwargs = {
    'pre_train': pre_train_fn,
    'test_params': {'timeout': 60, 'estimator__time_limit': 25 }
}

