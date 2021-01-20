import pytest
from glob import glob
from experiment.evaluate_model import evaluate_model
import importlib

# WARNING: this glob assumes tests are running from project root directory
MLs = [ml.split('/')[-1][:-3] for ml in glob('experiment/methods/*.py') if
       not ml.split('/')[-1][:-3].startswith('_')]
print('MLs:',MLs)

@pytest.mark.parametrize("ml", MLs)
def test_evaluate_model(ml):
    print('running test_evaluate_model with ml=',ml)
    dataset = 'experiment/test/192_vineyard_small.tsv.gz'
    results_path = 'tmp_results'
    random_state = 42

    algorithm = importlib.__import__('experiment.methods.'+ml,globals(),
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
                   True # testing
                  )
