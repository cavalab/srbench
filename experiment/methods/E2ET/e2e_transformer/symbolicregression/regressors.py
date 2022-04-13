from abc import ABC, abstractmethod
import numpy as np
import scipy
try: import xgboost as xgb
except: print("xgb problems")
from sympy import *

def order_data(X, y):
    idx_sorted = np.squeeze(np.argsort(X[:, :1], axis=0), -1)
    X = X[idx_sorted]
    y = y[idx_sorted]
    return X, y


def get_infinite_relative_error(prediction, truth):
    abs_relative_error = np.abs((prediction - truth) / (truth + 1e-100))
    abs_relative_error = np.nan_to_num(abs_relative_error, nan=np.infty)
    return np.max(abs_relative_error)


class Regressor(ABC):
    def __init__(self, **args):
        pass

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass


class DeepSymbolicRegressor(Regressor):
    def __init__(self):
        import dso

        self.model = dso.DeepSymbolicRegressor()
        self.n_samples = 2000

    def fit(self, X, y):
        if len(y.shape)>1: y = np.squeeze(y,-1)
        assert X.shape[0] == y.shape[0]
        self.model.fit(X, y, n_samples=self.n_samples)

    def predict(self, X):
        return self.model.predict(X)

class LagrangeRegressor(Regressor):
    def __init__(self):
        pass

    def fit(self, X, y):
        if len(y.shape)>1: y = np.squeeze(y,-1)

        assert X.shape[0] == y.shape[0]
        # X,y = order_data(X,y)
        X = np.squeeze(X, -1)
        self.model = scipy.interpolate.lagrange(X, y)

    def predict(self, X):
        if getattr(self, "model", None) is None:
            assert False
        X = np.squeeze(X, -1)
        return self.model(X)


class CubicSplineRegressor(Regressor):
    def __init__(self):
        pass

    def fit(self, X, y):
        if len(y.shape)>1: y = np.squeeze(y,-1)
        assert X.shape[0] == y.shape[0]
        X, y = order_data(X, y)

        X = np.squeeze(X, -1)
        diff = np.diff(X)
        duplicates = np.concatenate([[False], diff == 0])
        X = X[~duplicates]
        y = y[~duplicates]
        self.model = scipy.interpolate.CubicSpline(X, y)

    def predict(self, X):
        if getattr(self, "model", None) is None:
            assert False
        X = np.squeeze(X, -1)
        return self.model(X)


class gplearnSymbolicRegressor(Regressor):
    def __init__(self, function_set, const_range):
        import gplearn.genetic

        self.admissible_function_set = {
            'add': lambda x, y : x + y,
            'sub': lambda x, y : x - y,
            'mul': lambda x, y : x*y,
            'div': lambda x, y : x/y,
            'sqrt': lambda x : x**0.5,
            'log': lambda x : log(x),
            'abs': lambda x : abs(x),
            'neg': lambda x : -x,
            'inv': lambda x : 1/x,
            'max': lambda x, y : max(x, y),
            'min': lambda x, y : min(x, y),
            'sin': lambda x : sin(x),
            'cos': lambda x : cos(x),
            'pow': lambda x, y : x**y,
        }

        self.function_set = function_set.split(",")
        self.function_set = list(set(self.function_set) & set(self.admissible_function_set.keys()))
        self.const_range = const_range
        self.model = gplearn.genetic.SymbolicRegressor(
            function_set=self.function_set, const_range=self.const_range
        )

    def get_function(self):
        return sympify(str(self.model._program), locals=self.admissible_function_set)

    def fit(self, X, y):
        if len(y.shape)>1: y = np.squeeze(y,-1)
        assert X.shape[0] == y.shape[0]
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


class LinearRegressor(Regressor):
    def __init__(self):
        import sklearn.linear_model

        self.model = sklearn.linear_model.LinearRegression()

    def fit(self, X, y):
        if len(y.shape)>1: y = np.squeeze(y,-1)
        assert X.shape[0] == y.shape[0]
        self.model.fit(X, y)

    def predict(self, X):
        try: 
            return self.model.predict(X)
        except Exception as e:
            print(e, X)
            return None
    
class MLPRegressor(Regressor):
    def __init__(self):
        import sklearn.neural_network

        self.model = sklearn.neural_network.MLPRegressor()

    def fit(self, X, y):
        if len(y.shape)>1: y = np.squeeze(y,-1)
        assert X.shape[0] == y.shape[0]
        self.model.fit(X, y)

    def predict(self, X):
        try: 
            return self.model.predict(X)
        except Exception as e:
            print(e, X)
            return None
    
class XGBoostRegressor(Regressor):
    def __init__(self,**params):
        self.params=params
        pass

    def fit(self, X, y):
        if len(y.shape)>1: y = np.squeeze(y,-1)
        assert X.shape[0] == y.shape[0]
        self.model = xgb.XGBRegressor(**self.params)
        try:
            self.model.fit(X,y)
        except Exception as e:
            self.model = None

    def predict(self, X):
        if getattr(self, "model", None) is None:
            return None
        return self.model.predict(X)
