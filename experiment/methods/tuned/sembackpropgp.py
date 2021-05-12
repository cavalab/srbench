from ..sembackpropgp import complexity, model, est
from pyGPGOMEA import GPGOMEARegressor as GPG
from .params._sembackpropgp import params

est.set_params(**params)
est.functions = '+_-_*_aq_plog_sin_cos'

#double the evals
est.evaluations=1000000
est.time = 8*60*60
