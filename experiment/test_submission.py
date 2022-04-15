import pytest
import sys
import os
import json
import types
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

# shared variables
dataset = 'test/192_vineyard_small.tsv.gz'
results_path = 'tmp_results'
random_state = 42

def get_algorithm(ml):

    algorithm = importlib.__import__('methods.'+ml+'.regressor',globals(),
                                     locals(),
                                     ['*'])
    print('algorithm imported:',algorithm)
    return algorithm

def test_import(ml):
    """Check algorithm imports"""

    algorithm = get_algorithm(ml)
    assert 'est' in dir(algorithm)
    assert 'model' in dir(algorithm)

    eval_kwargs = {}
    if 'eval_kwargs' in dir(algorithm):
        eval_kwargs = algorithm.eval_kwargs
        eval_kwarg_types = {
            'test_params':dict,
            'max_train_samples':int,
            'scale_x':bool,
            'scale_y':bool,
            'pre_train':types.FunctionType
        }
        for k,v in eval_kwargs.items():
            assert k in ['test_params', 
                         'max_train_samples', 
                         'scale_x', 
                         'scale_y',
                         'pre_train'
                        ]
            assert isinstance(v, eval_kwarg_types[k])


def test_evaluate(ml):
    """Dataset evaluation"""
    algorithm = get_algorithm(ml)
    if 'eval_kwargs' in dir(algorithm):
        eval_kwargs = algorithm.eval_kwargs
    

    json_file = evaluate_model(dataset, 
                   results_path, 
                   random_state, 
                   ml,
                   algorithm.est, 
                   algorithm.model,
                   test=True, # testing
                   **eval_kwargs
                  )
    print(json_file)
    print("hello")


@pytest.mark.order(after="test_evaluate")
def test_sympy(ml):
    """Sympy compatibility of model string"""
    dataset_name = dataset.split('/')[-1][:-7]
    json_file = (results_path + '/' + dataset_name + '_' + ml + '_' 
                 + str(random_state) + '.json')
    if os.path.exists(json_file):
        r = json.load(open(json_file, 'r'))
    else:
        raise FileNotFoundError(json_file+' not found')

    est_name = r['algorithm']

    raw_model = r['symbolic_model']
    print('raw_model:',raw_model)
    X, labels, features = read_file(dataset)
    local_dict = {k:Symbol(k) for k in features}

    model_sym = parse_expr(raw_model, local_dict = local_dict)
    model_sym = round_floats(model_sym)
    print('sym model:',model_sym)

    model_complexity = complexity(model_sym)
    print('model complexity:',model_complexity)