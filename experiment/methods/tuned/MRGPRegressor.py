from ..src.mrgp import MRGPRegressor
from .params._mrgpregressor import params

est=MRGPRegressor(max_len=6,
                  time_out=10*60*60)

est.set_params(**params)

# double the evals
est.g *= 2**0.5
est.popsize *= 2**0.5

def complexity(est):
    return est.complexity

def model(est):
    return str(est.model_)
