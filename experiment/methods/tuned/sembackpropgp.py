from ..sembackpropgp import complexity as complexity
from ..sembackpropgp import est as est
from ..sembackpropgp import model as model
from .params._sembackpropgp import params

est.set_params(**params)
est.functions = "+_-_*_aq_plog_sin_cos"

# double the evals
est.evaluations = 1000000
est.time = 8 * 60 * 60
