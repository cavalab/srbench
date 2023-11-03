import sys
from os.path import dirname as d
from os.path import abspath
root_dir = d(d(abspath(__file__)))
sys.path.append(root_dir)
print('appended',root_dir,'to sys.path')

from evaluate_model import evaluate_model
import importlib

def test_evaluate_model(ml):
    print('running test_evaluate_model with ml=',ml)
    dataset = 'test/192_vineyard_small.tsv.gz'
    results_path = 'tmp_results'
    random_state = 42

    algorithm = importlib.__import__(f'methods.{ml}.regressor',globals(),
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
