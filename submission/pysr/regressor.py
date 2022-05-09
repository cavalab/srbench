# This example submission shows the submission of FEAT (cavalab.org/feat).
from pysr import PySRRegressor
import re
import sympy
import os

num_cores = os.environ["OMP_NUM_THREADS"]
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
    procs=num_cores,
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
    est.reset()


# define eval_kwargs.
eval_kwargs = dict(
    pre_train=my_pre_train_fn, test_params=dict(populations=5, timeout_in_seconds=60)
)
