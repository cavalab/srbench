#from ITEA 
import sys
import os
os.environ["LD_LIBRARY_PATH"] = os.environ["CONDA_PREFIX"] + "/lib"
from pyITEA import ITEARegressor
from itertools import product

hyper_params = [
    {
        'exponents' : ((0,5),),
        'termlimit' : ((2,5),),
        'transfunctions' : ('[Id, Sin]',)
    },
    {
        'exponents' : ((0,5),),
        'termlimit' : ((2, 15),),
        'transfunctions' : ('[Id, Sin]',)
    },
    {
        'exponents' : ((-5,5),),
        'termlimit' : ((2, 15),),
        'transfunctions' : ('[Id, Sin]',)
    },
    {
        'exponents' : ((-5, 5),),
        'termlimit' : ((2, 5),),
        'transfunctions' : ('[Id, Tanh, Sin, Cos, Log, Exp, SqrtAbs]',)
    },
    {
        'exponents' : ((0, 5),),
        'termlimit' : ((2, 15),),
        'transfunctions' : ('[Id, Tanh, Sin, Cos, Log, Exp, SqrtAbs]',)
    },
    {
        'exponents' : ((-5, 5),),
        'termlimit' : ((2, 15),),
        'transfunctions' : ('[Id, Tanh, Sin, Cos, Log, Exp, SqrtAbs]',)
    },
]

# Create the pipeline for the model
eval_kwargs = {'scale_x': False, 'scale_y': False}
est = ITEARegressor(npop=1000, ngens=500, exponents=(-1, 1), termlimit=(2,
    2), nonzeroexps=1, 
    transfunctions= '[Id, Tanh, Sin, Cos, Log, Exp, SqrtAbs]'
    )

def complexity(e):
    return e.len

def model(e):
    return e.expr
