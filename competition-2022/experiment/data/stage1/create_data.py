from distutils import dir_util
from sklearn.metrics import r2_score
import sklearn.datasets as datasets
import pandas as pd
import numpy as np
from numpy.random import default_rng
from scipy.stats import pearsonr
import math

import create_data_exact_formula as exact

# directory = r'datasets/synthetic/data/'
directory='data/'
seeds = [2082, 3946, 5847, 6087, 6589, 7102, 7884, 8674, 9023, 9900]

# todo add different seeds


def main():
    for seed in seeds:
        generate_data(seed)


def generate_data(seed: int):
# WGL
    exact_train, exact_test = exact.generate_exact_formula_data_easier(seed)
    save_data(exact_train, exact_test, f'{seed}_exact_formula_easier')

    exact_train, exact_test = exact. generate_exact_formula_data_easy(seed)
    save_data(exact_train, exact_test, f'{seed}_exact_formula_easy')
    exact_train, exact_test = exact. generate_exact_formula_data_medium(seed)
    save_data(exact_train, exact_test, f'{seed}_exact_formula_medium')
    exact_train, exact_test = exact. generate_exact_formula_data_hard(seed)
    save_data(exact_train, exact_test, f'{seed}_exact_formula_hard')

    # extrapolation_train, extrapolation_test = generate_extrapolation_data(seed, noise_ratio=0.025)
    # save_data(extrapolation_train, extrapolation_test, f'{seed}_extrapolation_easy')
    # extrapolation_train, extrapolation_test = generate_extrapolation_data(seed, noise_ratio=0.05)
    # save_data(extrapolation_train, extrapolation_test, f'{seed}_extrapolation_medium')
    # extrapolation_train, extrapolation_test = generate_extrapolation_data(seed, noise_ratio=0.1)
    # save_data(extrapolation_train, extrapolation_test, f'{seed}_extrapolation_hard')

    # feature_train, feature_test = generate_featureselection_data(seed, noise_ratio=0.025)
    # save_data(feature_train, feature_test, f'{seed}_featureselection_easy')
    # feature_train, feature_test = generate_featureselection_data(seed, noise_ratio=0.05)
    # save_data(feature_train, feature_test, f'{seed}_featureselection_medium')
    # feature_train, feature_test = generate_featureselection_data(seed, noise_ratio=0.1)
    # save_data(feature_train, feature_test, f'{seed}_featureselection_hard')

    # noise_train, noise_test = generate_noise_data(seed, noise_ratio=0.0)
    # save_data(noise_train, noise_test, f'{seed}_noise_base')
    # noise_train, noise_test = generate_noise_data(seed, noise_ratio=0.05)
    # save_data(noise_train, noise_test, f'{seed}_noise_easy')
    # noise_train, noise_test = generate_noise_data(seed, noise_ratio=0.1)
    # save_data(noise_train, noise_test, f'{seed}_noise_medium')
    # noise_train, noise_test = generate_noise_data(seed, noise_ratio=0.15)
    # save_data(noise_train, noise_test, f'{seed}_noise_hard')

    # localopt_train, localopt_test = generate_localoptima_data(seed, num_surrogate_features=3, noise_ratio=0.1)
    # save_data(localopt_train, localopt_test, f'{seed}_localopt_easy')
    # localopt_train, localopt_test = generate_localoptima_data(seed, num_surrogate_features=4, noise_ratio=0.1)
    # save_data(localopt_train, localopt_test, f'{seed}_localopt_medium')
    # localopt_train, localopt_test = generate_localoptima_data(seed, num_surrogate_features=5, noise_ratio=0.1)
    # save_data(localopt_train, localopt_test, f'{seed}_localopt_hard')


def save_data(data_train: pd.DataFrame, data_test: pd.DataFrame, name: str, directory: str = directory):
    data = pd.concat([data_train, data_test])
    data.to_csv(directory+name+'_data.csv', index=False)
    data_train.to_csv(directory+name+'_data_train.csv', index=False)
    data_test.to_csv(directory+name+'_data_test.csv', index=False)


def generate_extrapolation_data(seed: int, noise_ratio: float = 0.0):
    # https://www.desmos.com/calculator/hgsi4wi01n
    def func(x):
        return math.erf(0.22 * x) + 0.17 * math.sin(5.5 * x)

    x = np.arange(start=-15.0, stop=40.01, step=0.1)
    y = np.array([func(xi) for xi in x])

    df = pd.DataFrame({'x': x, 'y': y})
    if noise_ratio != 0.0:
        add_noise(df, target='y', noise_ratio=noise_ratio, seed=seed)
        df = df.drop(columns=['y'])
        df.columns.values[-1] = 'y'

    df_train = df[df['x'] <= 15]
    df_test = df[df['x'] > 15]
    return df_train, df_test


def generate_localoptima_data(seed: int, num_surrogate_features: int = 2, noise_ratio: float = 0.01):
    num_atomic_features = 5
    samples = 1000
    train = 100
    rng = np.random.default_rng(seed)

    def basefun(x):
        y = 0
        return y

    def subfun1(x):
        y = 0.77 * x[0]*x[1]
        return y

    def subfun2(x):
        y = 1.52*x[1] * x[2]
        return y

    def subfun3(x):
        y = 1.2*x[3]**2
        return y

    def subfun4(x):
        y = 0.31 * x[0] * x[3] * x[4]
        return y

    def subfun5(x):
        y = 0.23 * x[2] * x[3] * x[4]
        return y

    funs = [subfun1, subfun2, subfun3, subfun4, subfun5]

    def func(x):
        y = basefun(x)
        for f in funs[:num_surrogate_features]:
            y += f(x)
        return y

    num_features = num_atomic_features + num_surrogate_features
    x = rng.uniform(-5.0, 5.0, size=(samples, num_atomic_features))

    x_surr = list()
    for f in funs[:num_surrogate_features]:
        surr_feat = [f(r) for r in x]
        sigma_noise = np.std(surr_feat) * math.sqrt(noise_ratio / (1.0 - noise_ratio))
        surr_feat += rng.normal(loc=0.0, scale=sigma_noise, size=len(x))
        x_surr.append(surr_feat)

    x_surr = np.array(x_surr, dtype=float) .transpose()
    xx = np.hstack((x, x_surr))

    columns = ['x' + str(i+1) for i in range(num_features)]
    y = [func(r) for r in x]

    df = pd.DataFrame(xx, columns=columns)
    df['y'] = y

    df_train = df.iloc[0:train]
    df_test = df.iloc[train:]

    return df_train, df_test


def generate_featureselection_data(seed: int, noise_ratio: float = 0.0,):
    features = 20
    samples = 1000
    train = 500
    rng = np.random.default_rng(seed)

    def func(x): 
        y = 0.11 * x[0]**3
        y += 0.91 * x[2]*x[4]
        y += 0.68 * x[6]*x[8]
        y += 0.26 * x[10]**2 * x[12]
        y += 0.13 * x[14]*x[16]*x[18]
        return y

    x = rng.uniform(-10.0, 10.0, size=(samples, features))
    columns = ['x' + str(x) for x in range(1, features + 1)]
    y = [func(r) for r in x]

    df = pd.DataFrame(x, columns=columns)
    df['y'] = y

    if noise_ratio != 0.0:
        add_noise(df, target='y', noise_ratio=noise_ratio, seed=seed)
        df = df.drop(columns=['y'])
        df.columns.values[-1] = 'y'

    # add_noise(df, target='y', noise_ratio=0.025, seed=seed)
    # add_noise(df, target='y', noise_ratio=0.05, seed=seed)

    df_train = df.iloc[0:train]
    df_test = df.iloc[train:]

    return df_train, df_test


def generate_noise_data(seed: int, noise_ratio: float = 0.0):
    features = 1
    samples = 1000

    rng = np.random.default_rng(seed)

    def func(x):
        y = 0.11 * x**4
        y -= 1.4 * x**3
        y /= 0.68 * x**2 + 1
        return y

    x_train = np.arange(start=-10.0, stop=10.01, step=0.4)
    train = x_train.size
    x_test = rng.uniform(-10.0, 10.0, size=(samples-train, features))
    x_test = x_test.flatten()
    x = np.concatenate([x_train, x_test])

    columns = ['x' + str(x) for x in range(1, features + 1)]
    y = [func(r) for r in x]

    df = pd.DataFrame(x, columns=columns)
    df['y'] = y

    if noise_ratio != 0.0:
        add_noise(df, target='y', noise_ratio=noise_ratio, seed=seed)
        df = df.drop(columns=['y'])
        df.columns.values[-1] = 'y'

    df_train = df.iloc[0:train]
    df_test = df.iloc[train:]

    return df_train, df_test


def add_noise(df: pd.DataFrame, noise_ratio: float, seed: int = 0, target: str = None):
    if(noise_ratio < 0):
        raise ValueError("noise must be between 0 and 1.")
    if(noise_ratio > 1):
        raise ValueError("noise must be between 0 and 1.")

    if(target == None):
        target_series = df.iloc[:, -1]
    else:
        target_series = df[target]
    sigma_noise = target_series.std() * math.sqrt(noise_ratio / (1.0 - noise_ratio))

    rng = default_rng(seed=seed)
    nd_values = rng.normal(loc=0.0, scale=sigma_noise, size=len(target_series))

    df[target_series.name +
       f' Noise {noise_ratio}'] = target_series + nd_values

    return df


if __name__ == "__main__":
    main()
