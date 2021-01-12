import pytest
import os
from glob import glob
from experiment.evaluate_model import evaluate_model
import importlib

# WARNING: this glob assumes tests are running from project root directory
MLs = [ml.split('/')[-1][:-3] for ml in glob('experiment/methods/*.py')]
print('MLs:',MLs)
#TODO: change this to a call to evaluate_model so we can separate tests
@pytest.mark.parametrize("ml", MLs)
def test_evaluate_model(ml):
    print('running test_evaluate_model with ml=',ml)
    dataset = 'experiment/test/192_vineyard_small.tsv.gz'
    results_path = 'tmp_results'
    random_state = 42

    algorithm = importlib.__import__('experiment.methods.'+ml,globals(),
                                     locals(),
                                   ['est','hyper_params','complexity'])

    evaluate_model(dataset, 
                   results_path, 
                   random_state, 
                   ml,
                   algorithm.est, 
                   algorithm.hyper_params, 
                   algorithm.complexity,
                   True
                  )

# def test_analyze_runs_with_all_methods():
#     """Each method runs on a small dataset"""
#     f = 'tests/192_vineyard_small.tsv.gz'

#     ml_files = glob('./methods/*.py')
#     mls = ','.join([m[10:-3] for m in ml_files if '__' not in m])
#     print('MLs:',mls)

#     jobline =  ('python analyze.py {DATA} '
#                '-ml {ML} '
#                '-results {RDIR} -n_trials {NT} -n_jobs 1 --local -test').format(
#                    DATA=f,
#                    ML=mls,
#                    RDIR='tmp_results',
#                    NT=1)
#     print(jobline)
#     os.system(jobline)

# if __name__ == '__main__':
#     test_analyze_runs_with_all_methods()
