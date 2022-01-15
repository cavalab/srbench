from pysr import pysr, best_row
from multiprocessing import cpu_count
from functools import partial
import numpy as np


## All PySR settings:
# binary_operators=None,
# unary_operators=None,
# procs=4,
# loss="L2DistLoss()",
# populations=20,
# niterations=100,
# ncyclesperiteration=300,
# alpha=0.1,
# annealing=False,
# fractionReplaced=0.10,
# fractionReplacedHof=0.10,
# npop=1000,
# parsimony=1e-4,
# migration=True,
# hofMigration=True,
# shouldOptimizeConstants=True,
# topn=10,
# weightAddNode=1,
# weightInsertNode=3,
# weightDeleteNode=3,
# weightDoNothing=1,
# weightMutateConstant=10,
# weightMutateOperator=1,
# weightRandomize=1,
# weightSimplify=0.01,
# perturbationFactor=1.0,
# timeout=None,
# extra_sympy_mappings=None,
# extra_torch_mappings=None,
# extra_jax_mappings=None,
# equation_file=None,
# verbosity=1e9,
# progress=None,
# maxsize=20,
# fast_cycle=False,
# maxdepth=None,
# variable_names=None,
# batching=False,
# batchSize=50,
# select_k_features=None,
# warmupMaxsizeBy=0.0,
# constraints=None,
# useFrequency=True,
# tempdir=None,
# delete_tempfiles=True,
# julia_optimization=3,
# julia_project=None,
# user_input=True,
# update=True,
# temp_equation_file=False,
# output_jax_format=False,
# output_torch_format=False,
# optimizer_algorithm="BFGS",
# optimizer_nrestarts=3,
# optimize_probability=1.0,
# optimizer_iterations=10,
# tournament_selection_n=10,
# tournament_selection_p=1.0,
# denoise=False,
# Xresampled=None,
# precision=32,
# multithreading=None,


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
    return est.get_best()['sympy_format']


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
