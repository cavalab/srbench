from bingo.symbolic_regression.symbolic_regressor import SymbolicRegressor, CrossValRegressor
from bingo.evolutionary_algorithms.age_fitness import AgeFitnessEA

from sklearn.model_selection import KFold

hyper_params = [
    # narrowed to 3 by looking at most commonly chosen
    # among previous SRBench run
    {"population_size": [100], "stack_size": [24]},
    {"population_size": [500], "stack_size": [24]},
    {"population_size": [2500], "stack_size": [32]}
]

"""
est: a sklearn-compatible regressor.
"""
non_tuned_est = SymbolicRegressor(population_size=500, stack_size=24,
                                  operators=["+", "-", "*", "/",
                                             "sin", "cos", "exp", "log",
                                             "sqrt"],
                                  use_simplification=True,
                                  crossover_prob=0.3, mutation_prob=0.45, metric="mse",
                                  parallel=False, clo_alg="lm", max_time=350, max_evals=int(1e19),
                                  evolutionary_algorithm=AgeFitnessEA,
                                  clo_threshold=1.0e-5)


N_FOLDS = 3

# use `N_FOLDS` grid search cross-validation to select hyper parameters
cv = KFold(n_splits=N_FOLDS, shuffle=True)

est = CrossValRegressor(non_tuned_est, cv=cv, param_grid=hyper_params,
                        verbose=3, n_jobs=1, scoring="r2", error_score="raise")


def model(est, X=None):
    """
    Return a sympy-compatible string of the final model. 

    Parameters
    ----------
    est: sklearn regressor
        The fitted model. 
    X: pd.DataFrame, default=None
        The training data. This argument can be dropped if desired.

    Returns
    -------
    A sympy-compatible string of the final model. 

    Notes
    -----

    Ensure that the variable names appearing in the model are identical to 
    those in the training data, `X`, which is a `pd.Dataframe`. 
    If your method names variables some other way, e.g. `[x_0 ... x_m]`, 
    you can specify a mapping in the `model` function such as:

        ```
        def model(est, X):
            mapping = {'x_'+str(i):k for i,k in enumerate(X.columns)}
            new_model = est.model_
            for k,v in mapping.items():
                new_model = new_model.replace(k,v)
        ```
    """
    model_str = str(est.get_best_individual())

    try:
        # replace X_# with data variables names
        mapping = {'X_' + str(i): k for i, k in enumerate(X.columns)}
        for k,v in mapping.items():
            model_str = model_str.replace(k,v)
    except AttributeError:  # if X is not a pd.Dataframe
        pass

    model_str = model_str.replace(")(", ")*(").replace("^", "**")  # replace operators for sympy
    return model_str


# for calculating time per cross-validation run using
# grid search, not halving grid search!
def get_cv_time(total_time):
    return total_time / (len(hyper_params) * N_FOLDS + 1)  # have to train each hyperparam
    # set on `N_FOLDS` then also need to retrain final model


def pre_train_fn(est, X, y):
    """set max_time in seconds based on length of X."""
    if len(X) <= 1000:
        max_time = get_cv_time(60 * 60 - 100)  # 1 hour with 100 seconds of slack
    else:
        max_time = get_cv_time(10 * 60 * 60 - 1000)  # 10 hours with 1000 seconds of slack
    est.set_max_time(new_max_time=max_time)


eval_kwargs = {
    "pre_train": pre_train_fn
}

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

