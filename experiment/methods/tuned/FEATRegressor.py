from ..FEATRegressor import complexity,model,pre_train, eval_kwargs
from feat import FeatRegressor
from .params._featregressor import params

est = FeatRegressor(
                    pop_size=500,
                    gens=200,
                    max_time=2*60*60,  # 2 hrs
                    max_stall=100,
                    batch_size=100,
                    max_depth=6,
                    max_dim=10,
                    backprop=True,
                    iters=1,
                    n_jobs=1,
                    simplify=0.005,
                    corr_delete_mutate=True,
                    cross_rate=0.75,
                    verbosity=0
                   )
est.set_params(**params)
print('blah')
est.functions = '+,-,*,/,^2,^3,sqrt,sin,cos,exp,log' 
# double the evals
est.gens *= 2**0.5
est.pop_size *= 2**0.5
