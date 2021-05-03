from .src.mrgp import MRGPRegressor


# 500,000 evaluations = 250,000 with 1 linear regression iteration
pop_sizes = [100, 500, 1000]
gs = [2500, 500, 250]
rt_xos = [0.2, 0.8]
hyper_params = []

for p, g in zip(pop_sizes, gs):
    for rt_xo in rt_xos:
        hyper_params.append(
               {'popsize':[p],
                'g':[g],
                'rt_cross':[rt_xo],
                'rt_mut':[1-rt_xo]
                }
               )
est=MRGPRegressor(max_len=6,
                  time_out=8*60*60
        )

def complexity(est):
    return est.complexity

def model(est):
    return str(est.model_)
