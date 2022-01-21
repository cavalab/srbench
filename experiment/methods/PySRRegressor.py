from pysr import pysr, best_row
from multiprocessing import cpu_count
from functools import partial
import numpy as np

# Import base estimator and regressor mixin:
from sklearn.base import BaseEstimator, RegressorMixin


class PySRRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        model_selection="accuracy",
        binary_operators=None,
        unary_operators=None,
        populations=40,
        niterations=20,
        ncyclesperiteration=300,
        annealing=False,
        maxsize=20,
        npop=100,
        parsimony=0.0,
        procs=cpu_count(),
    ):
        super().__init__()
        self.model_selection = model_selection

        # PySR parameters:

        if binary_operators is None:
            binary_operators = ["+", "-", "*", "/"]
        if unary_operators is None:
            unary_operators = []

        self.binary_operators = binary_operators
        self.unary_operators = unary_operators

        self.populations = populations
        self.niterations = niterations
        self.ncyclesperiteration = ncyclesperiteration
        self.annealing = annealing
        self.procs = procs
        self.maxsize = maxsize
        self.npop = npop
        self.parsimony = parsimony

        # Stored equations:
        self.equations = None

    def __repr__(self):
        return f"PySRRegressor(equations={self.get_best()['sympy_format']})"

    def get_best(self):
        if self.equations is None:
            return dict(
                sympy_format=0.0,
                complexity=1,
                lambda_format=lambda x: 0.0,
            )
        if self.model_selection == "accuracy":
            return self.equations.iloc[-1]
        elif self.model_selection == "best":
            return best_row(self.equations)
        else:
            raise NotImplementedError

    def fit(self, X, y):
        self.equations = pysr(
            X=X,
            y=y,
            binary_operators=self.binary_operators,
            unary_operators=self.unary_operators,
            populations=self.populations,
            niterations=self.niterations,
            ncyclesperiteration=self.ncyclesperiteration,
            annealing=self.annealing,
            procs=self.procs,
            maxsize=self.maxsize,
            npop=self.npop,
            parsimony=self.parsimony,
            extra_sympy_mappings={
                "square": lambda x: x ** 2,
                "cube": lambda x: x ** 3,
                "quart": lambda x: x ** 4,
            },
        )
        return self

    def predict(self, X):
        equation_row = self.get_best()
        np_format = equation_row["lambda_format"]

        return np_format(X)

    def get_params(self, deep=True):
        del deep
        return {
            "model_selection": self.model_selection,
            "binary_operators": self.binary_operators,
            "unary_operators": self.unary_operators,
            "populations": self.populations,
            "niterations": self.niterations,
            "ncyclesperiteration": self.ncyclesperiteration,
            "annealing": self.annealing,
            "procs": self.procs,
            "maxsize": self.maxsize,
            "npop": self.npop,
            "parsimony": self.parsimony,
        }


def complexity(est):
    return est.get_best()["complexity"]


def model(est):
    return str(est.get_best()["sympy_format"])


est = PySRRegressor()
poly_basis = ["square(x) = x^2", "cube(x) = x^3", "quart(x) = x^4"]
trig_basis = ["cos", "sin"]
exp_basis = ["exp", "log", "sqrt"]
hyper_params = [
    {
        "annealing": (True, False),
        "denoise": (True, False),
        "binary_operators": (["+", "-", "*", "/"],),
        "unary_operators": (
            [],
            poly_basis,
            poly_basis + trig_basis,
            poly_basis + exp_basis,
        ),
        "populations": (40, 80),
        "alpha": (0.01, 0.1, 1.0, 10.0),
        "model_selection": ("accuracy", "best"),
    }
]
