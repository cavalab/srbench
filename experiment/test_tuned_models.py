import sys
from os.path import dirname as d
from os.path import abspath, join
root_dir = d(d(abspath(__file__)))
sys.path.append(root_dir)
print('appended',root_dir,'to sys.path')

import pytest
from glob import glob
from evaluate_model import evaluate_model
import importlib

TMLs = ['tuned.'+ml.split('/')[-1][:-3] for ml in glob('methods/tuned/*.py') if
       not ml.split('/')[-1][:-3].startswith('_')]


@pytest.fixture()
def ml(pytestconfig):
    return pytestconfig.getoption("ml")

def test_tuned_models(ml):
    if ml not in TMLs:
        # Some algorithms not tuned.
        exit(0)

    print('running test_evaluate_model with ml=',ml)
    dataset = 'test/strogatz_shearflow1.tsv.gz'
    results_path = 'tmp_results'
    random_state = 42

    algorithm = importlib.__import__('methods.'+ml,
                                     globals(),
                                     locals(),
                                     ['*']
                                    )

    print('algorithm:',algorithm.est)
    if 'hyper_params' not in dir(algorithm):
        algorithm.hyper_params = {}
    print('hyperparams:',algorithm.hyper_params)

    # optional keyword arguments passed to evaluate
    eval_kwargs = {}
    if 'eval_kwargs' in dir(algorithm):
        eval_kwargs = algorithm.eval_kwargs

    # check for conflicts btw cmd line args and eval_kwargs
    # if args.SYM_DATA:
    eval_kwargs['scale_x'] = False
    eval_kwargs['scale_y'] = False
    eval_kwargs['skip_tuning'] = True
    eval_kwargs['sym_data'] = True

    evaluate_model(dataset, 
                   results_path, 
                   random_state, 
                   ml,
                   algorithm.est, 
                   algorithm.hyper_params, 
                   algorithm.complexity,
                   algorithm.model,
                   target_noise=0, 
                   feature_noise=0,
                   test=True, # testing
                   **eval_kwargs
                  )
