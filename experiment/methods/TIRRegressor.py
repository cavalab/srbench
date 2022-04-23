import sys
import os
os.environ["LD_LIBRARY_PATH"] = os.environ["CONDA_PREFIX"] + "/lib"
import pyTIR as tir
from itertools import product

hyper_params = [
    {
        'transfunctions' : ('Id,Tanh,Sin,Cos,Log,Exp,Sqrt',),
        'ytransfunctions' : ('Id,Sqrt,Exp,Log,ATan,Tan,Tanh',),
        'exponents' : ((-5,5),)
    },
    {
        'transfunctions' : ('Id,Tanh,Sin,Cos,Log,Exp,Sqrt',),
        'ytransfunctions' : ('Id,Sqrt,Exp,Log,ATan,Tan,Tanh',),
        'exponents' : ((0,5),)
    },
    {
        'transfunctions' : ('Id,Tanh,Sin,Cos,Log,Exp,Sqrt',),
        'ytransfunctions' : ('Id,Sqrt,Exp,Log,ATan,Tan,Tanh',),
        'exponents' : ((-1,1),)
    },
    {
        'transfunctions' : ('Id,Tanh,Sin,Cos,Log,Exp,Sqrt',),
        'ytransfunctions' : ('Id,Sqrt,Exp,Log,ATan,Tan,Tanh',),
        'exponents' : ((0,1),)
    },
    {
        'transfunctions' : ('Id,Tanh,Sin,Cos,Log,Exp,Sqrt',),
        'ytransfunctions' : ('Id,Sqrt,Exp,Log,ATan,Tan,Tanh',),
        'exponents' : ((-2,2),)
    },
    {
        'transfunctions' : ('Id,Tanh,Sin,Cos,Log,Exp,Sqrt',),
        'ytransfunctions' : ('Id,Sqrt,Exp,Log,ATan,Tan,Tanh',),
        'exponents' : ((0,2),)
    },
]
# Create the pipeline for the model

est = tir.TIRRegressor(npop=1000, ngens=500, pc=0.3, pm=0.7, exponents=(-5,5), error="R^2")

def pre_train(est, X, y):
    """Adjust settings based on data before training"""
    if X.shape[0]*X.shape[1] <= 1000:
        est.penalty = 0.01
    print('TIR penalty adjusted to',est.penalty)
            
def complexity(e):
    return e.len

def model(e):
    return e.expr

eval_kwargs = {'scale_x': False, 'scale_y': False, 'pre_train': pre_train}
