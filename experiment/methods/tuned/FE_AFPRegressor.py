from ..FE_AFPRegressor import complexity as complexity
from ..FE_AFPRegressor import est as est
from ..FE_AFPRegressor import eval_kwargs as eval_kwargs
from ..FE_AFPRegressor import model as model
from .params._fe_afpregressor import params

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
est.time_limit = 8 * 60 * 60
# doubling evals
est.g = int(est.g * 2**0.5)
est.popsize = int(est.popsize * 2**0.5)
