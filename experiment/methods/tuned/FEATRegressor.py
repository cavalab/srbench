from ..FEATRegressor import complexity as complexity
from ..FEATRegressor import est as est
from ..FEATRegressor import eval_kwargs as eval_kwargs
from ..FEATRegressor import model as model
from .params._featregressor import params

est.set_params(**params)

est.functions = "+,-,*,/,^2,^3,sqrt,sin,cos,exp,log"
est.otype = "f"
# double the evals
est.max_time = int(8 * 60 * 60)  # 8 hrs
est.gens = int(est.gens * 2**0.5)
est.pop_size = int(est.pop_size * 2**0.5)
