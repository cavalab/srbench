import pandas as pd
import numpy as np
from numpy.random import default_rng
import math


def generate_exact_formula_data_easier(seed: int = 0):
    features = 10
    samples = 2000
    train = 1000

    rng = np.random.default_rng(seed)
    x = rng.uniform(-5.0, 5.0, size=(samples, features))

    def func(x):
        y = 0
        y += 0.4 * x[0] * x[1]
        y += -1.5 * x[0] + 2.5 * x[1] + 1
        # y += math.log(x[2])
        # y += math.log(30 * x[2]**2)
        return y

    return create_dataframes(x, func, train)
def generate_exact_formula_data_easy(seed: int = 0):
    features = 10
    samples = 2000
    train = 1000

    rng = np.random.default_rng(seed)
    x = rng.uniform(-5.0, 5.0, size=(samples, features))

    def func(x):
        y = 0
        y += 0.4 * x[0] * x[1]
        y += -1.5 * x[0] + 2.5 * x[1] + 1
        y += math.log(30 * x[2]**2)
        return y

    return create_dataframes(x, func, train)


def generate_exact_formula_data_medium(seed: int = 0):
    features = 5
    samples = 2000
    train = 1000

    rng = np.random.default_rng(seed)
    x = rng.uniform(-5.0, 5.0, size=(samples, features))

    def func(x):
        y = 0
        y += 0.4 * x[0] * x[1]
        y += -1.5 * x[0] + 2.5 * x[1] + 1
        y /= 1 + 0.2 * (x[0]**2 + x[1]**2)
        return y

    return create_dataframes(x, func, train)


def generate_exact_formula_data_hard(seed: int = 0):
    features = 2
    samples = 2000
    train = 1000

    rng = np.random.default_rng(seed)
    x = rng.uniform(-5.0, 5.0, size=(samples, features))

    def func(x):
        y = 5.5*math.sin(x[0] + x[1])
        y += 0.4 * x[0] * x[1]
        y += -1.5 * x[0] + 2.5 * x[1] + 1
        y /= 1 + 0.2 * (x[0]**2 + x[1]**2)
        return y

    return create_dataframes(x, func, train)


def create_dataframes(x, func, train):
    features = x.shape[1]
    columns = ['x' + str(x) for x in range(1, features + 1)]
    y = [func(r) for r in x]

    df = pd.DataFrame(x, columns=columns)
    df['y'] = y

    df_train = df.iloc[0:train]
    df_test = df.iloc[train:]

    return df_train, df_test
