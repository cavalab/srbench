import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr

"""
Timeout handling
"""

MAXTIME = 60

import signal
class SimplifyTimeOutException(Exception):
    pass

def alarm_handler(signum, frame):
    print(f"raising SimplifyTimeOutException")
    raise SimplifyTimeOutException


"""
For all of these, higher is better
"""
def accuracy(est, X, y):
    pred = est.predict(X)

    mse = mean_squared_error(y, pred)
    y_var = np.var(y)
    if(y_var == 0.0):
        y_var = 1e-9

    r2 = 1 - mse / y_var
    r2 = np.round(r2, 3)
    return r2

"""
Utilities
"""


def round_floats(ex1):
    ex2 = ex1
    for a in sp.preorder_traversal(ex1):
        if isinstance(a, sp.Float):
            if abs(a) < 0.0001:
                ex2 = ex2.subs(a, sp.Integer(0))
            else:
                ex2 = ex2.subs(a, round(a, 3))
    return ex2



def get_symbolic_model(pred_model, local_dict):
    # TODO: update namespace for exact_formula runs
    sp_model = sp.parse_expr(pred_model, local_dict=local_dict)
    sp_model = round_floats(sp_model)

    signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(MAXTIME) # maximum time, defined above
    try:
        sp_model = sp.simplify(sp_model)
    except Exception as e:
        print('Warning: simplify failed. Msg:',e)
        pass
    return sp_model

def simplicity(pred_model, feature_names):
    local_dict = {f:sp.Symbol(f) for f in feature_names} 
    sp_model = get_symbolic_model(pred_model, local_dict)
    # compute num. components
    num_components = 0
    for _ in sp.preorder_traversal(sp_model):
        num_components += 1
    # compute simplicity as per juding criteria
    simplicity = -np.round(np.log(num_components)/np.log(5), 1)
    return simplicity

"""
Problem specific
"""
def symbolic_equivalence(true_model, pred_model, local_dict):
    """Check whether symbolic model is equivalent to the ground truth model."""
    sp_model = get_symbolic_model(pred_model, local_dict)

    sym_diff = round_floats(true_model - sp_model)
    sym_frac = round_floats(sp_model/true_model)
    print('true_model:',true_model, '; \npred_model:',sp_model)

    try:
        diff_const=sym_diff.is_constant(simplify=False) 
        frac_const=sym_frac.is_constant(simplify=False) 

        # check if we can skip simplification
        if not diff_const and not frac_const:
            signal.signal(signal.SIGALRM, alarm_handler)
            signal.alarm(MAXTIME) # maximum time, defined above
            try:
                if not diff_const:
                    sym_diff = sp.simplify(sym_diff)
                    diff_const=sym_diff.is_constant() 
                if not frac_const:
                    sym_frac = sp.simplify(sym_frac)
                    frac_const=sym_frac.is_constant() 
            except Exception as e:
                print('Warning: simplify failed. Msg:',e)
                pass
    except Exception as e:
        print('const checking failed.')
        diff_const=False
        frac_const=False
        pass


    result = dict(
            equivalent = (
                str(sym_diff) == '0'
                or diff_const 
                or frac_const
            ),
            sym_diff = str(sym_diff),
            sym_frac = str(sym_frac),
            true_model = str(true_model),
            pred_model = str(sp_model)
            )

    print('result:',result)
    return result
