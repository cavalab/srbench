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

print('blah2')

def complexity(est):
    complexity = est.get_n_nodes() + 2*est.get_n_params() + 2*est.get_dim() 
    return complexity

def model(est):
    return est.get_eqn()

def pre_train(est, X, y):
    """Adjust settings based on data before training"""
    # adjust generations based onsize of X versus batch size
    if est.batch_size < len(X):
        est.gens = int(est.gens*len(X)/est.batch_size)
    print('FEAT gens adjusted to',est.gens)
    # adjust max dim
    est.max_dim=min(max(est.max_dim, X.shape[1]), 20)
    print('FEAT max_dim set to',est.max_dim)

eval_kwargs = dict(pre_train=pre_train)
