from pysr import PySRRegressor
from multiprocessing import cpu_count
from functools import partial
import numpy as np

# Import base estimator and regressor mixin:
from sklearn.base import BaseEstimator, RegressorMixin


def complexity(est):
    return est.get_best().complexity


def model(est):
    return est.get_best().equation


est = PySRRegressor()
poly_basis = ["square(x) = x^2", "cube(x) = x^3", "quart(x) = x^4"]
trig_basis = ["cos", "sin"]
exp_basis = ["exp", "log", "sqrt"]
# Hyperparams are reduced for speed of testing:
hyper_params = [
    {
        "annealing": (True,), # (True, False)
        "denoise": (True,), # (True, False)
        "binary_operators": (["+", "-", "*", "/"],),
        "unary_operators": (
            [],
            # poly_basis,
            # poly_basis + trig_basis,
            # poly_basis + exp_basis,
        ),
        "populations": (20,), # (40, 80),
        "alpha": (1.0,),
        "model_selection": ("best",)
        # "alpha": (0.01, 0.1, 1.0, 10.0),
        # "model_selection": ("accuracy", "best"),
    }
]
