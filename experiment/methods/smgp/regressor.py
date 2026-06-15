# Scikit-like regressor wrapper around the symbolic genetic programming algorithm.
# This module exposes a fit/predict interface and configures the GP training process.

from __future__ import annotations

import time
import random
from copy import deepcopy
from typing import Optional, Union, List

import numpy as np
import importlib

import sys
import os

print(f"DEBUG: CWD (pracovní adresář): {os.getcwd()}")
print(f"DEBUG: __file__ (cesta k regressor.py): {os.path.abspath(__file__)}")
print(f"DEBUG: sys.path (kde Python hledá moduly): {sys.path}")
print(f"DEBUG: Obsah složky s regressor.py: {os.listdir(os.path.dirname(os.path.abspath(__file__)))}")

try:
    sklearn_base = importlib.import_module("sklearn.base")
    BaseEstimator = getattr(sklearn_base, "BaseEstimator")
    RegressorMixin = getattr(sklearn_base, "RegressorMixin")
    validation = importlib.import_module("sklearn.utils.validation")
    check_X_y = getattr(validation, "check_X_y")
    check_array = getattr(validation, "check_array")
except Exception:
    BaseEstimator = type("BaseEstimator", (), {})
    RegressorMixin = type("RegressorMixin", (), {})

    def check_X_y(X, y, dtype=None):
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError("X must be a 2D array or matrix")
        if y_arr.ndim != 1:
            raise ValueError("y must be a 1D array")
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        return X_arr, y_arr

    def check_array(X, dtype=None):
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError("X must be a 2D array or matrix")
        return X_arr

from smgp.crossover import Crossover
from smgp.data import ArrayDataSource
from smgp.evolution import VectorEvolutionAlgorithm
from smgp.fitness import MeanSquaredErrorFitnessFunctionVector
from smgp.individual import Individual
from smgp.mutation import Mutation
from smgp.smoothMultifunctionSet import SmoothMultifunctionSet
from smgp.variable import Variable


class SMGPRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        random_state: Optional[int] = None,
        max_time: Optional[float] = None,
        population_size: int = 100,
        generations: int = 100000,
        depth: int = 6,
        mutation_rate: float = 0.03,
        random_individual_rate: float = 0.08,
        variable_probability: float = 0.45,
        min_terminal_node_val: float = 0.0,
        max_terminal_node_val: float = 10.0,
        function_set: Optional[Union[SmoothMultifunctionSet, List[str]]] = None,
        taylor_sum_elements: int = 5,
        use_triangle_fval: bool = True,
        verbose: bool = False,
    ):
        self.random_state = random_state
        self.max_time = max_time
        self.population_size = population_size
        self.generations = generations
        self.depth = depth
        self.mutation_rate = mutation_rate
        self.random_individual_rate = random_individual_rate
        self.variable_probability = variable_probability
        self.min_terminal_node_val = min_terminal_node_val
        self.max_terminal_node_val = max_terminal_node_val
        self.taylor_sum_elements = taylor_sum_elements
        self.use_triangle_fval = use_triangle_fval
        self.function_set = function_set
        self.verbose = verbose

        self.feature_names_in_ = None
        self.n_features_in_ = None
        self.best_individual_ = None
        self.best_fitness_ = None
        self.model_ = None
        self.history_ = []
        self.rng = random.Random(self.random_state) if self.random_state is not None else random.Random()

    def _check_dataframe_columns(self, X):
        if hasattr(X, "columns"):
            return list(X.columns)
        return None

    def _prepare_random_state(self):
        if self.random_state is not None:
            random.seed(self.random_state)
            np.random.seed(self.random_state)
            self.rng = random.Random(self.random_state)
        else:
            self.rng = random.Random()
        
        # Configure or create the SmoothMultifunctionSet dynamically during fit
        if self.function_set is None:
            # Default to the Basic blackbox function set instead of Classic
            self.function_set = SmoothMultifunctionSet.createBasicFunctionSet(
                taylorSumElements=self.taylor_sum_elements, 
                useTriangleFval=self.use_triangle_fval
            )
        elif isinstance(self.function_set, list):
            # If function_set is a list of strings from SRBench, build it dynamically
            self.function_set = SmoothMultifunctionSet.createMultifunctionSetByNames(
                function_names=self.function_set,
                taylorSumElements=self.taylor_sum_elements,
                useTriangleFval=self.use_triangle_fval
            )
        elif isinstance(self.function_set, SmoothMultifunctionSet):
            # If already an object, update its Taylor parameters to match current wrapper configuration
            self.function_set.taylorSumElements = self.taylor_sum_elements
            self.function_set.useTriangleFval = self.use_triangle_fval

    def _create_variable_list(self, feature_names: List[str]) -> List[Variable]:
        return [Variable(name) for name in feature_names]

    def _fitness(self, individual: Individual, X: np.ndarray, y: np.ndarray, function_list: list, variable_list: List[Variable]) -> float:
        n_samples = X.shape[0]
        if n_samples == 0:
            raise ValueError("Training data must contain at least one sample.")

        total_error = 0.0
        for row_idx in range(n_samples):
            Variable.setVariableValues(variable_list, X[row_idx].tolist())
            predicted = individual.evaluate(function_list, variable_list)
            total_error += float((predicted - y[row_idx]) ** 2)

        mse = total_error / float(n_samples)
        if mse == 0.0:
            return float('inf')
        return 1.0 / mse

    def _evaluate(self, individual: Individual, X: np.ndarray, variable_list: List[Variable], function_list: list) -> np.ndarray:
        predictions = np.empty(X.shape[0], dtype=float)
        for i in range(X.shape[0]):
            Variable.setVariableValues(variable_list, X[i].tolist())
            predictions[i] = individual.evaluate(function_list, variable_list)
        return predictions

    def _function_list(self) -> list:
        return [self.function_set]

    def _to_sympy_string(self, individual: Individual, feature_names: List[str], fset: SmoothMultifunctionSet) -> str:
        max_val = fset.maxVal

        def op_str(idx_func: int, left: str, right: str) -> str:
            func_obj = fset.functionMap.get(idx_func)
            if func_obj is None:
                raise ValueError(f"Unsupported function index {idx_func}")
            
            name = func_obj.name

            if name == "+": return f"({left} + {right})"
            if name == "-": return f"({left} - {right})"
            if name == "*": return f"({left}*{right})"
            if name == "/": return f"({left}/{right})"
            if name == "^": return f"({left}**{right})"
            
            if name in ["sin", "cos", "tan", "exp", "log", "sqrt", "abs", 
                        "asin", "acos", "atan", "sinh", "cosh", "tanh"]:
                return f"{name}({left})"
                
            raise ValueError(f"Unknown operator name '{name}' at index {idx_func}")

        def node_expr(idx: int, depth: int) -> str:
            if depth >= individual.depth:
                value = individual.vector[2 * idx]
                type_flag = int(individual.vector[2 * idx + 1])
                if type_flag == 0:
                    return f"{float(value):.12g}"
                return feature_names[int(value)]

            left_expr = node_expr(2 * idx + 1, depth + 1)
            right_expr = node_expr(2 * idx + 2, depth + 1)
            
            fval = float(individual.vector[2 * idx])
            
            if fset.useTriangleFval:
                from taylorFunc import TaylorFunc
                val = TaylorFunc.x_triangle(fval, fset.taylorSumElements)
            else:
                val = fval

            func_idx = int(val)
            next_idx = (func_idx + 1) % max_val
            frac = val - func_idx

            first_expr = op_str(func_idx, left_expr, right_expr)
            if frac == 0.0:
                return first_expr
                
            second_expr = op_str(next_idx, left_expr, right_expr)
            return f"(({1 - frac:.12g})*{first_expr} + {frac:.12g}*{second_expr})"

        return node_expr(0, 1)

    def fit(self, X: Union[np.ndarray, List[List[float]], object], y: Union[np.ndarray, List[float]]) -> 'SMGPRegressor':
        X_arr, y_arr = check_X_y(X, y, dtype=[np.float64, np.float32])
        feature_names = self._check_dataframe_columns(X) or [f"x_{i}" for i in range(X_arr.shape[1])]
        self.feature_names_in_ = feature_names
        self.n_features_in_ = X_arr.shape[1]

        if self.n_features_in_ == 0:
            raise ValueError("At least one feature is required.")
        if self.generations < 1:
            raise ValueError("generations must be at least 1")
        if self.population_size < 1:
            raise ValueError("population_size must be at least 1")

        self._prepare_random_state()
        variable_list = self._create_variable_list(feature_names)
        function_list = [self.function_set]

        data_source = ArrayDataSource(X_arr, y_arr, feature_names)
        algorithm = VectorEvolutionAlgorithm(
            fList=function_list,
            dataSource=data_source,
            fitnessFunction=MeanSquaredErrorFitnessFunctionVector(),
            dataIndexes=list(range(X_arr.shape[0])),
            mutationFunc=Mutation.vectorMutation,
             crossoverFunc=Crossover.betweenPointCrossover,
            rng=self.rng,
            taylorSumElements=self.taylor_sum_elements,
            useTriangleFval=self.use_triangle_fval,
        )

        best_individual = algorithm.runEvolution(
            maxGenerations=self.generations,
            populationSize=self.population_size,
            depth=self.depth,
            mutationRate=self.mutation_rate,
            randomIndividualRate=self.random_individual_rate,
            variableProbability=self.variable_probability,
            minTerminalNodeVal=self.min_terminal_node_val,
            maxTerminalNodeVal=self.max_terminal_node_val,
            maxSeconds=self.max_time,
        )

        if best_individual is None:
            raise RuntimeError("No individual was found during fit.")

        self.best_individual_ = best_individual
        self.best_fitness_ = self._fitness(best_individual, X_arr, y_arr, function_list, variable_list)
        
        # Correctly passing the configured function_set object into the sympy transformation method
        self.model_ = self._to_sympy_string(best_individual, self.feature_names_in_, self.function_set)
        return self

    def predict(self, X: Union[np.ndarray, List[List[float]], object]) -> np.ndarray:
        if self.best_individual_ is None:
            raise ValueError("Estimator is not fitted yet. Call fit() before predict().")

        X_arr = check_array(X, dtype=[np.float64, np.float32])
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X_arr.shape[1]} features, but this estimator was fitted with {self.n_features_in_} features.")

        variable_list = self._create_variable_list(self.feature_names_in_)
        function_list = self._function_list()
        return self._evaluate(self.best_individual_, X_arr, variable_list, function_list)


def model(est: SMGPRegressor, X=None):
    model_str = getattr(est, 'model_', None)
    if model_str is None:
        raise ValueError('Estimator has no model_ string. Fit the estimator first.')
    if X is None:
        return model_str

    if hasattr(X, 'columns'):
        mapping = {f'x_{i}': name for i, name in enumerate(X.columns)}
        result = model_str
        for old_name, new_name in reversed(list(mapping.items())):
            result = result.replace(old_name, new_name)
        return result

    return model_str    