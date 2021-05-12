from ..OperonRegressor import complexity,model,est
from operon.sklearn import SymbolicRegressor
import operon._operon as op
from .params._operonregressor import params

est.set_params(**params)
est.allowed_symbols = 'add,mul,aq,exp,log,sin,tanh,constant,variable'

# double the evals
est.max_evaluations = 1000000
est.generations=100000, # just large enough since we have an evaluation budget
est.time_limit=8*60*60 # 8 hours
