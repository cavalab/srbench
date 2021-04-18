from sklearn import ensemble

hyper_params = [{
    'learning_rate' : (0.01, 0.1, 1.0, 10.0,),
    'n_estimators' : (10, 100, 1000,),
}]

est=ensemble.AdaBoostRegressor()

complexity = None
model = None
model = None
