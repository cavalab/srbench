from rils_rols.rils_rols import RILSROLSRegressor
from symbolic_utils import complexity as cplx
import sympy as sp

eval_kwargs = {'scale_x': False, 'scale_y': False}
# This is a setup for ground-truth instances (feynman and strogatz), for black-box just change max fit calls to 500k, i.e., max_fit_calls=500*1000,
# if there is still that requirement (as in paper), otherwise keep 1 million. 
est = RILSROLSRegressor(max_seconds=2*60*60, max_fit_calls=1000*1000, max_complexity=50, sample_size=0, verbose=True)

def model(est):
    return str(est.model_string())


def complexity(est):
    return cplx(sp.parse_expr(str(est.model_string())))
