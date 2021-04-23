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

# WARNING: this glob assumes tests are running from project root directory
MLs = [ml.split('/')[-1][:-3] for ml in glob('methods/FEAT*.py') if
       not ml.split('/')[-1][:-3].startswith('_')]
print('MLs:',MLs)

@pytest.mark.parametrize("ml", MLs)
def test_evaluate_model(ml):
    print('running test_evaluate_model with ml=',ml)
    dataset = 'test/192_vineyard_small.tsv.gz'
    results_path = 'tmp_results'
    random_state = 42

    algorithm = importlib.__import__('methods.'+ml,globals(),
                                     locals(),
                                   ['est','hyper_params','complexity'])

    print('algorithm imported:',algorithm)
    evaluate_model(dataset, 
                   results_path, 
                   random_state, 
                   ml,
                   algorithm.est, 
                   algorithm.hyper_params, 
                   algorithm.complexity,
                   algorithm.model,
                   test=True # testing
                  )
