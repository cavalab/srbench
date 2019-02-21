import sys
import xgboost
# read arguments (1: dataset name, 2: output file name, 3: trial #, 4: # cores)
dataset = sys.argv[1]
savefile = sys.argv[2]
random_state = sys.argv[3]

hyper_params = [
    {
        'alpha': (1e-06,1e-04,0.01,1,),
        'penalty': ('l2','l1','elasticnet',),
    },
]

hyper_params = [
    {
        'n_estimators' : (10, 50, 100, 250, 500, 1000,),
        'learning_rate' : (0.0001,0.01, 0.05, 0.1, 0.2,),
        'gamma' : (0,0.1,0.2,0.3,0.4,),
        'max_depth' : (6,),
        'subsample' : (0.5, 0.75, 1,),
    },
]

est=xgboost.XGBRegressor()


from .evaluate_model import evaluate_model
evaluate_model(dataset, save_file, random_state, est, hyper_params)
