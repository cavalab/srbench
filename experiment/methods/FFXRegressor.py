from ffx import FFXRegressor

hyper_params = [
    {
        # TODO: we could define a hyperparameter to tune how we choose
        # from the Pareto front. For now, FFXRegressor will just return
        # the most complex (= most accurate).
    },
]

est = FFXRegressor()

def complexity(est):
    return est.model_.complexity()

def model(est):
    return est.model_.str2()
