import xgboost

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

def complexity(est):
    return np.sum([m.count(':') for m in est._Booster.get_dump()])
