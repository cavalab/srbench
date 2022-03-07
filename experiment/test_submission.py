import sys
from os.path import dirname as d
from os.path import abspath
root_dir = d(d(abspath(__file__)))
sys.path.append(root_dir)
print('appended',root_dir,'to sys.path')

from evaluate_model import evaluate_model
import importlib
# symbolic model stuff
from sympy.parsing.sympy_parser import parse_expr
from symbolic_utils import (complexity, round_floats,
                            sub, div, square, cube, quart,
                            PLOG, PLOG10, PSQRT)
from read_file import read_file
from sympy import Symbol 


def test_submission(ml):
    print('running test_evaluate_model with ml=',ml)
    dataset = 'test/192_vineyard_small.tsv.gz'
    results_path = 'tmp_results'
    random_state = 42

    algorithm = importlib.__import__('methods.'+ml+'.regressor',globals(),
                                     locals(),
                                   ['est','hyper_params','complexity'])

    print('algorithm imported:',algorithm)
    json_file = evaluate_model(dataset, 
                   results_path, 
                   random_state, 
                   ml,
                   algorithm.est, 
                   algorithm.hyper_params, 
                   # algorithm.complexity,
                   algorithm.model,
                   test=True # testing
                  )

    ########################################
    # test sympy compatibility of model string
    if os.path.exists(json_file):
        r = json.load(open(json_file, 'r'))
    else:
        raise FileNotFoundError(json_file+' not found')

    est_name = r['algorithm']

    raw_model = r['symbolic_model']
    X, labels, features = read_file(dataset)
    local_dict = {k:Symbol(k) for k in features}

    model_sym = parse_expr(raw_model, local_dict = local_dict)
    model_sym = round_floats(model_sym)
    print('sym model:',model_sym)

    model_complexity = complexity(model_sym)
    print('model complexity:',model_complexity)


