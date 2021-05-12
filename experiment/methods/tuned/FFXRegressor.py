from ..FFXRegressor import complexity,model
from ffx import FFXRegressor

# FFX has no parameters!
hyper_params = [
    {
        # TODO: we could define a hyperparameter to tune how we choose
        # from the Pareto front. For now, FFXRegressor will just return
        # the most complex (= most accurate).
    },
]

est = FFXRegressor()
