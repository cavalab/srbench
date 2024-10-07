import sys
import os
import pyTIR as tir
from itertools import product
#os.environ["LD_LIBRARY_PATH"] = os.environ["CONDA_PREFIX"] + "/lib"

hyper_params = [
    {
        'transfunctions' : ('Id,Tanh,Sin,Cos,Log,Exp,Sqrt',),
        'ytransfunctions' : ('Id,Sqrt,Exp,Log,ATan,Tan,Tanh',),
        'exponents' : ((-5,5),)
    },
    {
        'transfunctions' : ('Id,Tanh,Sin,Cos,Log,Exp,Sqrt',),
        'ytransfunctions' : ('Id,Sqrt,Exp,Log,ATan,Tan,Tanh',),
        'exponents' : ((-2,2),)
    },
]

# Create the pipeline for the model
eval_kwargs = {'scale_x': False, 'scale_y': False}
est = tir.TIRRegressor(npop=1000, ngens=500, pc=0.3, pm=0.7, exponents=(-5,5), error="R^2", alg="MOO")

def pre_train(est, X, y):
    """Adjust settings based on data before training"""
    if X.shape[0]*X.shape[1] <= 1000:
        est.penalty = 0.01

def complexity(e):
    return e.len

def model(e, X):
    new_model = e.sympy.replace("^","**")
    for i,f in reversed(list(enumerate(X.columns))):
        new_model = new_model.replace(f'x{i}',f)
    return new_model

def get_population(est):
    pop = []
    for i in range(len(est.front)):
        pop.append(est.create_model_from(i))
    return pop 

def get_best_solution(est):
    return est
