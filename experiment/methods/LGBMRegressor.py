import lightgbm
import numpy as np

hyper_params = {
        'n_estimators' : (10, 50, 100, 250, 500, 1000,),
        'learning_rate' : (0.0001,0.01, 0.05, 0.1, 0.2,),
        'subsample' : (0.5, 0.75, 1,),
        'boosting_type' : ('gbdt', 'dart', 'goss')
    }

est=lightgbm.LGBMRegressor(
                           max_depth=6,
                           deterministic = True,
                           force_row_wise = True
                          )

def complexity(est):
    return np.sum([x['num_leaves'] for x in est._Booster.dump_model()['tree_info']])

model = None
