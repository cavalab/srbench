from .src.ITEA import itea
from itertools import product

es = [(-5, 5), (0, 5), (-2, 2), (0, 2)]
ts = [(2, 10), (2, 15), (2, 5)]
nzs = [1]
tss = [ '[Id]', '[Id, Tanh, Sin, Cos, Log, Exp, SqrtAbs]', '[Id, Sin]']

hyper_params = [
    {
        'exponents': (e,),
        'termlimit':(t,), 'nonzeroexps': (nz,),
        'transfunctions':(ts,),
    } for e, t, nz, ts in product(es, ts, nzs, tss)
]

# Create the pipeline for the model
eval_kwargs = {'scale_x': False, 'scale_y': False}
est = itea.ITEARegressor(npop=500, ngens=200, exponents=(-1, 1), termlimit=(2, 2))

def complexity(e):
    return e.len

def model(e):
    return e.expr
