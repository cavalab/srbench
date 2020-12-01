from sklearn import ensemble

hyper_params = [{
    'n_estimators': (10, 100, 1000),
    'min_weight_fraction_leaf': (0.0, 0.25, 0.5),
    'max_features': ('sqrt','log2',None),
}]


est=ensemble.RandomForestRegressor()

def complexity(est):
    return np.sum([e.tree_.node_count for e in est.estimators_])
