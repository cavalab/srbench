from pysr import pysr, best_row
from multiprocessing import cpu_count
from functools import partial
import numpy as np


class PySRRegressor:
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
        nprocs=cpu_count(),
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
        self.nprocs = nprocs
        self.maxsize = maxsize
        self.npop = npop
        self.parsimony = parsimony

        # Stored equations:
        self.equations = None

    def __repr__(self):
        return f"PySRRegressor(equations={self.get_best()['sympy_format']})"

    def get_best(self):
        if self.model_selection == "accuracy":
            return self.equations.iloc[-1]
        elif self.model_selection == "best":
            return best_row(self.equations)

    def fit(self, X, y):
        self.equations = self.pysr_call(
            X=X,
            y=y,
            binary_operators=self.binary_operators,
            unary_operators=self.unary_operators,
            populations=self.populations,
            niterations=self.niterations,
            ncyclesperiteration=self.ncyclesperiteration,
            annealing=self.annealing,
            nprocs=self.nprocs,
            maxsize=self.maxsize,
            npop=self.npop,
            parsimony=self.parsimony,
        )
        return self

    def predict(self, X):
        equation_row = self.get_best()
        np_format = equation_row["lambda_format"]

        return np_format(X)

def complexity(est):
    return est.get_best()['complexity']

def model(est):
    return str(est.get_best()['sympy_format'])


est = PySRRegressor()
hyper_params = [
    {
        "annealing": (True, False),
        "denoise": (True, False),
        "binary_operators": (["+", "-", "*", "/"],),
        "unary_operators": ([], ["cos", "sin"], ["exp", "log", "sqrt"])
        "populations": (40, 80),
        "alpha": (0.01, 0.1, 1.0, 10.0),
        "model_selection": ("accuracy", "best"),
    }
]
