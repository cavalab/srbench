from feat import FeatRegressor

# 500,000 evaluations = 250,000 with 1 backprop iteration
pop_sizes = [100, 500, 1000]
gs = [2500, 500, 250]
lrs = [0.1, 0.3]
hyper_params = []
for p, g in zip(pop_sizes, gs):
    for lr in lrs:
        hyper_params.append({
            'pop_size':[p],
            'gens':[g],
            'lr':[lr]
        })

est = FeatRegressor(
                    pop_size=500,
                    gens=200,
                    max_time=28800,  # 8 hrs
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
                    verbosity=1
                   )

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
