from ..MRGPRegressor import complexity as complexity
from ..MRGPRegressor import est as est
from ..MRGPRegressor import model as model
from .params._mrgpregressor import params

est.set_params(**params)

# double the evals
est.time_out = 8 * 60 * 60
est.g = int(est.g * 2**0.5)
est.popsize = int(est.popsize * 2**0.5)
