import lightgbm
import numpy as np

# original set of hyperparameters (too many for the experimental design):
# {
#     'n_estimators' : (10, 100, 1000),
#     'learning_rate' : (0.0001, 0.01),
#     'subsample' : (0.5, 0.75),
#     'boosting_type' : ('gbdt', 'dart', 'goss')
# }

hyper_params = []
for n_estimators in (10, 100, 1000):
    for learning_rate in (0.0001, 0.01):
        for boosting_type in ("gbdt", "dart", "goss"):
            hyper_params.append(
                {
                    "n_estimators": [n_estimators],
                    "learning_rate": [learning_rate],
                    "boosting_type": [boosting_type],
                }
            )

est = lightgbm.LGBMRegressor(max_depth=6, deterministic=True, force_row_wise=True)


def complexity(est):
    c = np.sum([x["num_leaves"] for x in est._Booster.dump_model()["tree_info"]])
    return int(c)


model = None
