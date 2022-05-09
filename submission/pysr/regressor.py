# This example submission shows the submission of FEAT (cavalab.org/feat).
from pysr import PySRRegressor
import os

num_cores = os.cpu_count()
# functions ='+,-,*,/,^2,^3,sqrt,sin,cos,exp,log',


def get_best_equation(est):
    """Custom metric to get the best expression.

    First we filter based on loss, then we take the best score.
    """
    equations = est.equations
    best_loss = equations.loss.min()
    filtered = equations.query(f"loss < {2 * best_loss}")
    return equations.iloc[filtered.score.idxmax()]


est = PySRRegressor(
    binary_operators=["+", "-", "*", "/"],
    unary_operators=[
        "square",
        "cube",
        "cos",
        "sin",
        "exp",
        "slog(x::T) where {T} = (x > 0) ? log(x) : T(-1e9)",
        "ssqrt(x::T) where {T} = (x >= 0) ? sqrt(x) : T(-1e9)",
    ],
    maxsize=30,
    maxdepth=20,
    populations=50,
    niterations=1000000,
    timeout_in_seconds=60 * 60 * 1,
    constraints={
        "square": 8,
        "cube": 8,
        "exp": 8,
        "slog": 8,
        "ssqrt": 8,
        "sin": 8,
        "cos": 8,
        "/": (-1, 9),
    },
    nested_constraints={
        "cos": {"cos": 0, "sin": 0},
        "sin": {"sin": 0, "cos": 0},
        "/": {"/": 1},
        "slog": {"slog": 0, "exp": 0},
        "ssqrt": {"ssqrt": 0},
        "exp": {"exp": 0, "slog": 1},
        "square": {"square": 1, "cube": 1},
        "cube": {"cube": 1, "square": 1},
    },
)
# want to tune your estimator? wrap it in a sklearn CV class.


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
    """
    model_str = (
        get_best_equation(est).equation.replace("slog", "log").replace("ssqrt", "sqrt")
    )
    return model_str


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


def my_pre_train_fn(est, X, y):
    # """In this example we adjust FEAT generations based on the size of X
    # versus relative to FEAT's batch size setting.
    # """
    # if est.batch_size < len(X):
    #     est.gens = int(est.gens * len(X) / est.batch_size)
    # print("FEAT gens adjusted to", est.gens)
    # # adjust max dim
    # est.max_dim = min(max(est.max_dim, X.shape[1]), 20)
    # print("FEAT max_dim set to", est.max_dim)
    ...


# define eval_kwargs.
eval_kwargs = dict(
    pre_train=my_pre_train_fn, test_params=dict(populations=5, timeout_in_seconds=60)
)
