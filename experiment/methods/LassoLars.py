import sys
from sklearn import linear_model

# read arguments (1: dataset name, 2: output file name, 3: trial #, 4: # cores)
dataset = sys.argv[1]
savefile = sys.argv[2]
random_state = sys.argv[3]

hyper_params = [
    {
        'alpha': (1e-04,0.001,0.01,0.1,1,),
    },
]


est=linear_model.LassoLars()

from .evaluate_model import evaluate_model
evaluate_model(dataset, save_file, random_state, est, hyper_params)
