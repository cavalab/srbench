from ..OperonRegressor import complexity as complexity
from ..OperonRegressor import est as est
from ..OperonRegressor import model as model
from .params._operonregressor import params

est.set_params(**params)
est.allowed_symbols = "add,sub,mul,div,exp,log,sin,cos,sqrt,square,constant,variable"

# double the evals
est.max_evaluations = 1000000
est.generations = 100000  # just large enough since we have an evaluation budget
est.time_limit = 8 * 60 * 60  # 8 hours
