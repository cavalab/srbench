import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr

"""
Timeout handling
"""

MAXTIME = 10

import signal
class SimplifyTimeOutException(Exception):
    pass

def sym_alarm_handler(signum, frame):
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


def round_floats(ex1, dec=3):
    ex2 = ex1
    for a in sp.preorder_traversal(ex1):
        if isinstance(a, sp.Float):
            if abs(a) < 10**(-dec):
                ex2 = ex2.subs(a, sp.Integer(0))
            else:
                ex2 = ex2.subs(a, round(a, dec))
    return ex2



def get_symbolic_model(pred_model, local_dict):
    # TODO: update namespace for exact_formula runs
    sp_model = sp.parse_expr(pred_model, local_dict=local_dict)

    signal.signal(signal.SIGALRM, sym_alarm_handler)
    signal.alarm(MAXTIME) # maximum time, defined above
    try:
        sp_model = sp.simplify(sp_model)
    except SimplifyTimeOutException:
        print('simplify timed out; skipping...') 
        pass
    except:
        print('simplify failed; skipping...') 
        pass
    signal.alarm(0)

    sp_model = round_floats(sp_model)
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

    sym_diff = true_model - sp_model
    sym_frac = sp_model/true_model
    print('true_model:',true_model, '; \npred_model:',sp_model)

    try:
        diff_const=sym_diff.is_constant(simplify=False) 
        frac_const=sym_frac.is_constant(simplify=False) 

        # check if we can skip simplification
        if not diff_const and not frac_const:
            signal.signal(signal.SIGALRM, sym_alarm_handler)
            signal.alarm(MAXTIME) # maximum time, defined above
            try:
                if not diff_const:
                    sym_diff = sp.simplify(sym_diff)
                    diff_const=sym_diff.is_constant() 
                if not frac_const:
                    sym_frac = sp.simplify(sym_frac)
                    frac_const=sym_frac.is_constant() 
            except SimplifyTimeOutException:
                print('couldnt simplify frac or diff') 
                pass
            except:
                print('couldnt simplify frac or diff') 
                pass
            signal.alarm(0)
    except Exception as e:
        print('diff and frac const checking failed.')
        diff_const=False
        frac_const=False
        pass


    result = dict(
            equivalent = (
                str(sp_model) != '0'
                and 
                (str(sym_diff) == '0'
                 or diff_const 
                 or frac_const
                )
            ),
            sym_diff = str(sym_diff),
            sym_frac = str(sym_frac),
            true_model = str(true_model),
            pred_model = str(sp_model)
            )

    print('result:',result)
    return result


def feature_absence_score(pred_model, shouldnt_use, local_dict):
    """Counts up the number of features appearing in the model that shouldn't
        - higher is better
        - 1: no bad variables used
        - 0: using all bad variables
    """
    sp_model = get_symbolic_model(pred_model, local_dict)
    print('pred_model:',sp_model)
    print('shouldnt_use:',shouldnt_use)

    misused = set()
    for a in sp.preorder_traversal(sp_model):
        if isinstance(a, sp.Symbol):
            if a in shouldnt_use:
                misused.add(a)
                # print('adding',a,'to misused')
            # else:
                # print(a,'not in shouldnt_use')
    print('misused:',misused)
    score = (len(shouldnt_use)-len(misused)) / len(shouldnt_use)
    print('feature absence score:',score)
    return score


"""
Stuff for symbolic equivalence 
"""
# define local namespace dictionary
SEIR_VARS = {}
for s in ['dS', 'dE', 'dI', 'dR', 
          'x_S', 'x_E', 'x_I', 'x_R', 'x_rzero', 'x_gamma',
                  'x_sigma', 'x_c', 'x_N', 'x_Aux']:
        SEIR_VARS.update({ s: sp.Symbol(s) })
seir_gts = {
    "dS": get_symbolic_model('-x_gamma*x_rzero*x_S*x_I/x_N',SEIR_VARS),
    "dE": get_symbolic_model('x_gamma*x_rzero*x_S*x_I/x_N - x_sigma*x_E', SEIR_VARS),
    "dI": get_symbolic_model('x_sigma*x_E-x_gamma*x_I+x_c*x_R*x_I/x_N', SEIR_VARS),
    "dR": get_symbolic_model('x_gamma*x_I-x_c*x_R*x_I/x_N', SEIR_VARS)
}

EF_VARS = {f'x{i}':sp.Symbol(f'x{i}') for i in range(5)}
exact_formula_synthetic_str = {
    "easier": '0.4 * x1 * x2 - 1.5 * x1 + 2.5 * x2 + 1',
    "easy": '0.4 * x1 * x2 - 1.5 * x1 + 2.5 * x2 + 1 + log(30 * x3**2)',
    "medium": '(0.4 * x1 * x2 - 1.5 * x1 + 2.5 * x2 + 1) / (1 + 0.2 * (x1**2 + x2**2))',
    "hard": '(5.5 * sin(x1 + x2) + 0.4 * x1 * x2 - 1.5 * x1 + 2.5 * x2 + 1) / (1 + 0.2 * (x1**2 + x2**2))'
}
exact_formula_synthetic = {k:get_symbolic_model(v, EF_VARS) 
                           for k,v in exact_formula_synthetic_str.items()
                          }

"""
Main entry point
"""

def problem_specific_score(problem_name: str, 
                           est, 
                           X_test, 
                           y_test=None, 
                           pred_model=None
                          ):
    """
    Scoring functions for specific problems
    """
    # TODO: test whether true model returns true
    if "seir" in problem_name:
        target = problem_name.split("_")[1].replace('seir','')
        return ('symbolic_equivalence',
                symbolic_equivalence(true_model=seir_gts[target],
                                     pred_model=pred_model,
                                     local_dict=SEIR_VARS
                                     )
                )
    elif "exact_formula" in problem_name:
        target = None
        if "easy" in problem_name:
            target = exact_formula_synthetic["easy"]
        elif "easier" in problem_name:
            target = exact_formula_synthetic["easier"]
        elif "medium" in problem_name:
            target = exact_formula_synthetic["medium"]
        elif "hard" in problem_name:
            target = exact_formula_synthetic["hard"]
        else:
            raise ValueError(f"Unrecognized problem difficulty for {problem_name}")

        return ('symbolic_equivalence',
                symbolic_equivalence(true_model=target, 
                                     pred_model=pred_model,
                                     local_dict=EF_VARS
                                    )
               )

    elif "localopt" in problem_name:
        tot_num_features = X_test.shape[1]
        features = [f'x{i+1}' for i in range(tot_num_features)]
        localopt_features = list()
        for i in range(5, tot_num_features):
            localopt_features.append(f"x{i+1}")
        bad_features = features[5:]
        print('bad features:',bad_features)
        localopt_vars = {}
        # bad_features = []
        # for i,f in enumerate(features):
        #     localopt_vars.update({ f:sp.Symbol(f)})
        #     if i >= 5:
        #         bad_features.append(localopt_vars[f])

        localopt_vars = { f:sp.Symbol(f) for f in features } 

        return ('feature_absence_score',
                feature_absence_score(pred_model, 
                                      sp.symbols(bad_features),
                                      localopt_vars
                                     )
               )

    elif "extrapolation" in problem_name:
        return ('accuracy', accuracy(est, X_test, y_test))

    elif "featureselection" in problem_name:
        tot_num_features = X_test.shape[1]
        features = [f'x{i+1}' for i in range(tot_num_features)]
        irrelevant_features = list()
        for i,f in enumerate(features):
            if i % 2 == 0:  # even are OK, odd are irrelevant
                continue
            irrelevant_features.append(f)
        local_dict = { f:sp.Symbol(f) for f in features } 
        return ('feature_absence_score',
                feature_absence_score(pred_model, 
                                     sp.symbols(irrelevant_features),
                                     local_dict
                                    )
               )

    elif "noise" in problem_name:
        return ('accuracy', accuracy(est, X_test, y_test))

    else:
        # raise ValueError(f"Unrecognized problem name {problem_name}")
        print(f"No extra score for {problem_name}")
        return None, None
