import sys
import os
import types
import numpy as np
from os.path import dirname as d
from os.path import abspath
from sklearn.model_selection import train_test_split

root_dir = d(abspath(__file__))
sys.path.append(root_dir)
print('appended',root_dir,'to sys.path')

import importlib
from read_file import read_file

if 'OMP_NUM_THREADS' not in os.environ.keys():
    os.environ['OMP_NUM_THREADS'] = '1'
if 'OPENBLAS_NUM_THREADS' not in os.environ.keys():
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
if 'MKL_NUM_THREADS' not in os.environ.keys():
    os.environ['MKL_NUM_THREADS'] = '1'


def test_population(ml):
    """Sympy compatibility of model string"""

    dataset = 'test/192_vineyard_small.tsv.gz'
    random_state = 42

    algorithm = importlib.__import__(f'methods.{ml}.regressor',globals(),
                                        locals(),
                                    ['est','hyper_params','complexity'])

    algorithm.get_population,

    features, labels, feature_names =  read_file(
        dataset, 
        use_dataframe=True
    )
    print('feature_names:',feature_names)

    # generate train/test split
    X_train, X_test, y_train, y_test = train_test_split(features, labels,
                                                    train_size=0.75,
                                                    test_size=0.25,
                                                    random_state=random_state)
    
    # Few samples to try to make it quick
    sample_idx = np.random.choice(np.arange(len(X_train)), size=10)
    
    y_train = y_train.iloc[sample_idx]
    X_train = X_train.iloc[sample_idx, :]

    algorithm.est.fit(X_train, y_train)

    if 'get_population' not in dir(algorithm):
        algorithm.get_population = lambda est: [est]
    if 'get_best_solution' not in dir(algorithm):
        algorithm.get_best_solution = lambda est: est

    population = algorithm.get_population(algorithm.est)

    best_model = algorithm.get_best_solution(algorithm.est)
    print(algorithm.model(best_model))
    print(algorithm.est.predict(X_train))

    # assert that population has at least 1 and no more than 100 individuals
    assert 1 <= len(population) <= 100, "Population size is not within the expected range"
    
    for p in population:
        print(algorithm.model(p))
        print(p.predict(X_train))
