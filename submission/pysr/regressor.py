# This example submission shows the submission of FEAT (cavalab.org/feat).
from pysr import PySRRegressor
import re
import sympy
import os
import numpy as np

try:
    num_cores = os.environ["OMP_NUM_THREADS"]
except KeyError:
    from multiprocessing import cpu_count

    num_cores = cpu_count()


def get_best_equation(est):
    """Custom metric to get the best expression.

    First we filter based on loss, then we take the best score.
    """
    equations = est.equations
    best_loss = equations.loss.min()
    filtered = equations.query(f"loss < {2 * best_loss}")
    return equations.iloc[filtered.score.idxmax()]


warmup_time_in_minutes = 5
custom_operators = [
    "slog(x::T) where {T} = (x > 0) ? log(x) : T(-1e9)",
    "ssqrt(x::T) where {T} = (x >= 0) ? sqrt(x) : T(-1e9)",
]
standard_operators = [
    "square",
    "cube",
    "cos",
    "sin",
    "exp",
]

est = PySRRegressor(
    procs=num_cores,
    progress=False,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=standard_operators + custom_operators,
    maxsize=30,
    maxdepth=20,
    populations=50,
    niterations=1000000,
    timeout_in_seconds=60 * (60 - warmup_time_in_minutes),
    constraints={
        "square": 8,
        "cube": 8,
        "exp": 8,
        "sin": 8,
        "cos": 8,
        "/": (-1, 9),
        "slog": 8,
        "ssqrt": 8,
        "square": 8,
        "cube": 8,
    },
    nested_constraints={
        "cos": {"cos": 0, "sin": 0, "slog": 0, "exp": 0},
        "sin": {"sin": 0, "cos": 0, "slog": 0, "exp": 0},
        "/": {"/": 1},
        "exp": {"exp": 0, "slog": 1, "ssqrt": 0, "sin": 0, "cos": 0},
        "square": {"square": 1, "cube": 1},
        "cube": {"cube": 1, "square": 1},
        "slog": {"slog": 0, "exp": 0},
        "ssqrt": {"ssqrt": 1, "exp": 1},
    },
    extra_sympy_mappings={
        "slog": sympy.log,
        "ssqrt": sympy.sqrt,
    },
)
# want to tune your estimator? wrap it in a sklearn CV class.


def find_parens(s):
    """Copied from https://stackoverflow.com/questions/29991917/indices-of-matching-parentheses-in-python"""
    toret = {}
    pstack = []

    for i, c in enumerate(s):
        if c == "(":
            pstack.append(i)
        elif c == ")":
            if len(pstack) == 0:
                raise IndexError("No matching closing parens at: " + str(i))
            toret[pstack.pop()] = i

    if len(pstack) > 0:
        raise IndexError("No matching opening parens at: " + str(pstack.pop()))

    return toret


def replace_prefix_operator_with_postfix(s, prefix, postfix_replacement):
    while re.search(prefix, s):
        parens_map = find_parens(s)
        # Find parentheses at start of prefix:
        start_model_str = re.search(prefix, s).span()[0]
        start_parens = re.search(prefix, s).span()[1]
        end_parens = parens_map[start_parens]
        s = (
            s[:start_model_str]
            + "("
            + s[start_parens : end_parens + 1]
            + postfix_replacement
            + ")"
            + s[end_parens + 1 :]
        )
    return s


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
    model_str = get_best_equation(est).equation
    # Replacements:
    # slog => log
    model_str = re.sub("slog", "log", model_str)
    # ssqrt => sqrt
    model_str = re.sub("ssqrt", "sqrt", model_str)
    # square(...) => (...)**2
    model_str = replace_prefix_operator_with_postfix(model_str, "square", "**2")
    # cube(x) => (x)**3
    model_str = replace_prefix_operator_with_postfix(model_str, "cube", "**3")

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
    # Restart train state:
    est.reset()

    nfeatures = X.shape[1]
    nrows = X.shape[0]
    max_features = 8
    max_rows = 2000
    do_feature_selection = nfeatures > max_features
    do_resampling = nrows > max_rows

    if do_feature_selection:
        select_k_features = max_features
    else:
        select_k_features = None

    if do_resampling:
        xmin = np.min(X, axis=0)
        xmax = np.max(X, axis=0)
        Xresampled = np.stack(
            [
                np.random.uniform(
                    xmin[i],
                    xmax[i],
                    size=max_rows,
                )
                for i in range(nfeatures)
            ],
            axis=1,
        )
        denoise = True
    else:
        Xresampled = None
        denoise = False

    est.set_params(
        Xresampled=Xresampled,
        denoise=denoise,
        select_k_features=select_k_features,
    )


# define eval_kwargs.
eval_kwargs = dict(
    pre_train=my_pre_train_fn,
    test_params=dict(
        populations=5,
        niterations=5,
        maxsize=10,
        population_size=30,
        nested_constraints={},
        constraints={},
        # unary_operators=[],
    ),
)
