from ..BSRRegressor import complexity,model
from bsr.bsr_class import BSR
from .params._bsrregressor import params

# initialize
est = BSR( alpha1= 0.4, alpha2= 0.4, beta= -1, disp=False, max_time=2*60*60)

est.set_params(**params)

est.max_time = 8*60*60
# double the evals
est.itrNum = int(est.itrNum*2**0.5)
est.val = int(est.val*2**0.5)
