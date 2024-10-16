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

    algorithm = importlib.__import__(f'methods.{ml}.regressor',
                                     globals(), locals(),
                                    ['est','hyper_params','complexity'])

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
    
    y_train = y_train[sample_idx]
    X_train = X_train.iloc[sample_idx, :]

    ##################################################
    # fit with max_time
    ##################################################
    MAXTIME = 3600 # in seconds
    if hasattr(algorithm.est, 'max_time'):
        algorithm.est.max_time = MAXTIME
        print('max time:',MAXTIME)
    else:
        print('max time not set')

    algorithm.est.fit(X_train.values, y_train)

    ##################################################
    # get best solution
    ##################################################
    if 'get_best_solution' not in dir(algorithm):
        algorithm.get_best_solution = lambda est: est
        print(f"{ml} does not implement get_best_solution")
    else:
        print(f"{ml} has get_best_solution method. Using implemented method")

    best_model = algorithm.get_best_solution(algorithm.est)
    print('Best model')
    print(algorithm.model(best_model, X_train))
    print(algorithm.est.predict(X_train.values))

    ##################################################
    # get population
    ##################################################
    if 'get_population' not in dir(algorithm):
        algorithm.get_population = lambda est: [est]
        print(f"{ml} does not implement get_population")
    else:
        print(f"{ml} has get_population method. Using implemented method")

    population = algorithm.get_population(algorithm.est)
    assert 1 <= len(population) <= 100, \
        "Population size is not within the expected range"
    
    for i, p in enumerate(population):
        print(f"Individual {i}")
        print(algorithm.model(p, X_train))
        print(p.predict(X_train.values))
