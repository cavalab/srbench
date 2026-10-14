from ..GPGOMEARegressor import complexity as complexity
from ..GPGOMEARegressor import est as est
from ..GPGOMEARegressor import model as model
from .params._gpgomearegressor import params

est.set_params(**params)

est.functions = "+_-_*_p/_plog_sqrt_sin_cos"

# double the evals
est.evaluations = 1000000
est.time = 8 * 60 * 60
