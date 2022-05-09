from pysr import PySRRegressor
from multiprocessing import cpu_count
from functools import partial
import numpy as np
import sympy

# Import base estimator and regressor mixin:
from sklearn.base import BaseEstimator, RegressorMixin


def get_best_equation(est):
    """Custom metric to get the best expression.

    First we filter based on loss, then we take the best score.
    """
    equations = est.equations
    best_loss = equations.loss.min()
    filtered = equations.query(f"loss < {2 * best_loss}")
    return equations.iloc[filtered.score.idxmax()]


def complexity(est):
    return get_best_equation(est).complexity


def model(est):
    return get_best_equation(est).equation.replace("slog", "log").replace("ssqrt", "sqrt")


est = PySRRegressor(
    max_evals=500000, extra_sympy_mappings={"slog": sympy.log, "ssqrt": sympy.sqrt}
)
trig_basis = ["cos", "sin"]
exp_basis = [
    "exp",
    # Re-define these to prevent negative input (the default
    # for PySR is to use log(abs(x)), but here we assume
    # normal use of log(x))
    "slog(x::T) where {T} = (x > 0) ? log(x) : T(-1e9)",
    "ssqrt(x::T) where {T} = (x >= 0) ? sqrt(x) : T(-1e9)",
]
# Hyperparams are reduced for speed of testing:
hyper_params = [
    {
        "denoise": (True,),  # (True, False)
        "binary_operators": (["+", "-", "*", "/"],),
        "unary_operators": (
            # trig_basis
            trig_basis
            + exp_basis,
        ),
        "nested_constraints": (
            # Make the reasonable assumption that there are no
            # nested cos(cos(x)), etc. Likewise, we assume
            # there will never by a 1/(.../(.../...)), but
            # 1/(.../...) might occur.
            {
                "cos": {"cos": 0, "sin": 0},
                "sin": {"sin": 0, "cos": 0},
                "/": {"/": 1},
                "slog": {"slog": 0, "exp": 0},
                "ssqrt": {"ssqrt": 0},
                "exp": {"exp": 0, "slog": 1},
            },
        ),
        "populations": (40,),  # (40, 80),
    }
]
