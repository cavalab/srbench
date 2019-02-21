import sys
from sklearn import ensemble

# read arguments (1: dataset name, 2: output file name, 3: trial #, 4: # cores)
dataset = sys.argv[1]
savefile = sys.argv[2]
random_state = sys.argv[3]

hyper_params = [{
    'n_estimators': (10, 100, 1000),
    'min_weight_fraction_leaf': (0.0, 0.25, 0.5),
    'max_features': ('sqrt','log2',None),
}]


est=ensemble.RandomForestRegressor()

from .evaluate_model import evaluate_model
evaluate_model(dataset, save_file, random_state, est, hyper_params)
