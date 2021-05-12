from ..OperonRegressor import complexity,model
from operon.sklearn import SymbolicRegressor
import operon._operon as op
from .params._operonregressor import params

est = SymbolicRegressor(
            local_iterations=5,
            generations=100000, # just large enough since we have an evaluation budget
            n_threads=1,
            random_state=None,
            )

est.set_params(**params)
est.allowed_symbols = 'add,mul,aq,exp,log,sin,tanh,constant,variable'

# double the evals
est.max_evaluations = 1000000
est.time_limit=8*60*60 # 2 hours
