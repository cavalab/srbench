import numpy as np
import xgboost

hyper_params = [
    {
        "n_estimators": (10, 100, 1000),
        "learning_rate": (0.0001, 0.01, 0.1),
        "gamma": (0, 0.1, 0.2, 0.3, 0.4),
        "subsample": (0.5, 0.75),
    },
]

est = xgboost.XGBRegressor(max_depth=6)


def complexity(est):
    c = np.sum([m.count(":") for m in est._Booster.get_dump()])
    return int(c)


model = None
