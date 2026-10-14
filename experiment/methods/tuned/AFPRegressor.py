from ..AFPRegressor import complexity as complexity
from ..AFPRegressor import est as est
from ..AFPRegressor import model as model
from .params._afpregressor import params

est.set_params(**params)
est.op_list = [
    "n",
    "v",
    "+",
    "-",
    "*",
    "/",
    "exp",
    "log",
    "2",
    "3",
    "sqrt",
    "sin",
    "cos",
]
# double the evals
est.g = int(est.g * 2**0.5)
est.popsize = int(est.popsize * 2**0.5)
est.time_limit = 8 * 60 * 60
