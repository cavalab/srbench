import sys
from sklearn import svm

# read arguments (1: dataset name, 2: output file name, 3: trial #, 4: # cores)
dataset = sys.argv[1]
savefile = sys.argv[2]
random_state = sys.argv[3]

hyper_params = [
    {
        'C': (1e-06,1e-04,0.1,1,),
        'loss' : ('epsilon_insensitive','squared_epsilon_insensitive',),
    },
]

est=svm.LinearSVR()

from .evaluate_model import evaluate_model
evaluate_model(dataset, save_file, random_state, est, hyper_params)
