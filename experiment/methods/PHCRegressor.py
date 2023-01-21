
from HROCH import PHCRegressor
from symbolic_utils import complexity as srbench_complexity
import sympy as sp

# no hyperparameter tunning
hyper_params = [{}, ]

# scaling is not a good idea for symbolic methods
eval_kwargs = {'scale_x': False, 'scale_y': False}

# 1 minute limit to be comparable with nonsymbolic methods
est = PHCRegressor(time_limit=60.0, num_threads=1)


def model(est):
    return str(est.sexpr)


def complexity(est):
    return srbench_complexity(sp.parse_expr(est.sexpr))
