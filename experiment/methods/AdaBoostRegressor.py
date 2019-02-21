import sys
from sklearn import ensemble

# read arguments (1: dataset name, 2: save file name, 3: trial #)
dataset = sys.argv[1]
save_file = sys.argv[2]
random_state = sys.argv[3]

hyper_params = [{
    'learning_rate' : (0.01, 0.1, 1.0, 10.0,),
    'n_estimators' : (10, 100, 1000,),
}]

est=ensemble.AdaBoostRegressor()

from .evaluate_model import evaluate_model
evaluate_model(dataset, save_file, random_state, est, hyper_params)
