import sys
import xgboost
from sklearn.neural_network import MLPRegressor
# read arguments (1: dataset name, 2: output file name, 3: trial #, 4: # cores)
dataset = sys.argv[1]
savefile = sys.argv[2]
random_state = sys.argv[3]

hyper_params = [
    {
        'activation' : ('logistic', 'tanh', 'relu',),
        'solver' : ('lbfgs','adam','sgd',),
        'learning_rate' : ('constant', 'invscaling', 'adaptive',),
    },
]

est=MLPRegressor()

from .evaluate_model import evaluate_model
evaluate_model(dataset, save_file, random_state, est, hyper_params)

