from ..BSRRegressor import complexity,model,est
from .params._bsrregressor import params

est.set_params(**params)

est.max_time = 8*60*60
# double the evals
est.itrNum = int(est.itrNum*2**0.5)
est.val = int(est.val*2**0.5)
