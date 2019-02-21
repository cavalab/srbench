import sys
from sklearn import kernel_ridge

# read arguments (1: dataset name, 2: output file name, 3: trial #, 4: # cores)
dataset = sys.argv[1]
savefile = sys.argv[2]
random_state = sys.argv[3]

hyper_params = [{
    'kernel': ('linear', 'poly','rbf','sigmoid',),
    'alpha': (1e-4,1e-2,0.1,1,),
    'gamma': (0.01,0.1,1,10,),
}]

est=kernel_ridge.KernelRidge()


from .evaluate_model import evaluate_model
evaluate_model(dataset, save_file, random_state, est, hyper_params)
