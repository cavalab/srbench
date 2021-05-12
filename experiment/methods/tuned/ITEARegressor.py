from ..ITEARegressor import complexity,model,eval_kwargs
from .src.ITEA import itea_srbench as itea
from itertools import product
from .params._itearegressor import params

# Create the pipeline for the model
est = itea.ITEARegressor(npop=1000,
                         ngens=500,
                         exponents=(-1, 1),
                         termlimit=(2, 2),
                         nonzeroexps=1
                        )

est.set_params(**params)
est.transfunctions = '[Id, Tanh, Sin, Cos, Log, Exp, SqrtAbs]'

# double the evals
est.npop *= 2**0.5
est.ngens *= 2**0.5
