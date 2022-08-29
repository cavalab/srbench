import sys
import itertools
import pandas as pd
from sklearn.base import clone
from sklearn.experimental import enable_halving_search_cv # noqa
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  
import warnings
import time
from tempfile import mkdtemp
from shutil import rmtree
from joblib import Memory
from read_file import read_file
import pdb
import numpy as np
import json
import os
import inspect
from utils import jsonify
from symbolic_utils import get_sym_model

from metrics.evaluation import problem_specific_score, simplicity

import signal
class TimeOutException(Exception):
    pass

def alarm_handler(signum, frame):
    print(f"raising TimeOutException")
    raise TimeOutException

def set_env_vars(n_jobs):
    os.environ['OMP_NUM_THREADS'] = n_jobs 
    os.environ['OPENBLAS_NUM_THREADS'] = n_jobs 
    os.environ['MKL_NUM_THREADS'] = n_jobs

def evaluate_model(dataset, 
                   results_path, 
                   random_state, 
                   est_name, 
                   est, 
                   model, 
                   test=False, 
                   stage=0,
                   ##########
                   # valid options for eval_kwargs
                   ##########
                   test_params={},
                   max_train_samples=0, 
                   scale_x = False, 
                   scale_y = False,
                   pre_train=None,
                   DataFrame=True
                  ):

    print(40*'=','Evaluating '+est_name+' on ',dataset,40*'=',sep='\n')

    np.random.seed(random_state)
    if hasattr(est, 'random_state'):
        est.random_state = random_state

    ##################################################
    # setup data
    ##################################################
    idx={}
    X_train, X_test, y_train, y_test, idx['train'], idx['test'], feature_names, = \
            read_file(dataset, DataFrame=DataFrame, stage=stage,
                    random_state=random_state)

    # time limits
    if len(y_train) > 1000:
        MAXTIME = 36000
    else:
        MAXTIME = 3600

    print('max time:',MAXTIME)

    # if dataset is large, subsample the training set 
    if max_train_samples > 0 and len(y_train) > max_train_samples:
        print('subsampling training data from',len(X_train),
              'to',max_train_samples)
        sample_idx = np.random.choice(np.arange(len(X_train)),
                                      size=max_train_samples)
        y_train = y_train[sample_idx]
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.loc[sample_idx]
        else:
            X_train = X_train[sample_idx]

    # scale and normalize the data
    if scale_x:
        print('scaling X')
        sc_X = StandardScaler() 
        X_train_scaled = sc_X.fit_transform(X_train)
        X_test_scaled = sc_X.transform(X_test)
        if DataFrame:
            X_train_scaled = pd.DataFrame(X_train_scaled, 
                                          columns=feature_names)
            X_test_scaled = pd.DataFrame(X_test_scaled, 
                                          columns=feature_names)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test

    if not DataFrame: 
        assert isinstance(X_train_scaled, np.ndarray)
        assert isinstance(X_test_scaled, np.ndarray)

    if scale_y:
        print('scaling y')
        sc_y = StandardScaler()
        y_train_scaled = sc_y.fit_transform(y_train.reshape(-1,1)).flatten()
    else:
        y_train_scaled = y_train


    # run any method-specific pre_train routines
    if pre_train:
        pre_train(est, X_train_scaled, y_train_scaled)

    # define a test mode using estimator test_params, if they exist
    if test and len(test_params) != 0:
        print('WARNINING: Running in TEST mode.')
        est.set_params(**test_params)

    

    ################################################## 
    # Fit models
    ################################################## 
    print('X_train:',X_train_scaled.shape)
    print('y_train:',y_train_scaled.shape)
    print('training',est)
    t0t = time.time()
    signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(MAXTIME) # maximum time, defined above
    try:
        est.fit(X_train_scaled, y_train_scaled)
    except TimeOutException:
        print('WARNING: fitting timed out')
    signal.alarm(0)

    time_time = time.time() - t0t
    print('Training time measure:', time_time)
    
    ##################################################
    # store results
    ##################################################
    dataset_name = dataset.split('/')[-1].split('.')[0]
    results = {
        'dataset':dataset_name,
        'algorithm':est_name,
        'params':jsonify(est.get_params()),
        'random_state':random_state,
        'time_time': time_time, 
    }

    # get the final symbolic model as a string
    print('fitted est:',est)
    # print(vars(est))
    # print(dir(est))
    if 'X' in inspect.signature(model).parameters.keys():
        if not isinstance(X_train_scaled, pd.DataFrame):
            X_df = pd.DataFrame(X_train_scaled, 
                                          columns=feature_names)
        else:
            X_df = X_train_scaled
        results['symbolic_model'] = model(est, X_df)
    else:
        results['symbolic_model'] = model(est)

    ##################################################
    # scores
    ##################################################

    for fold, target, X in  [ 
                             ['train', y_train, X_train_scaled], 
                             ['test', y_test, X_test_scaled]
                            ]:
        # if len(X.shape) != 2:
        #     X = np.asarray(X).reshape(-1,1)

        y_pred = np.asarray(est.predict(X)).reshape(-1,1)
        if scale_y:
            y_pred = sc_y.inverse_transform(y_pred)

        for score, scorer in [('mse',mean_squared_error),
                              ('mae',mean_absolute_error),
                              ('r2', r2_score)
                             ]:
            results[score + '_' + fold] = scorer(target, y_pred) 
        if stage == 2:
            results[f'y_true_{fold}'] = target.tolist()
            results[f'y_pred_{fold}'] = y_pred.flatten().tolist()
            results[f'idx_{fold}'] = idx[fold].tolist()
    
    ##################################################
    # simplicity
    results['simplicity'] = simplicity(results['symbolic_model'], 
                                       feature_names
                                      )

    ##################################################
    # problem-specific scores
    y_pred = np.asarray(est.predict(X_test_scaled)).reshape(-1,1)
    metric, score = problem_specific_score(dataset, est, X_test, y_test=y_test, 
                                           pred_model=results['symbolic_model']
            )
    if metric != None:
        results[metric] = score 

    if stage == 2:
        import sympy as sp
        from metrics.evaluation import get_symbolic_model
        local_dict = { f:sp.Symbol(f) for f in feature_names } 
        results['simplified_model'] = str(get_symbolic_model(
                results['symbolic_model'],
                local_dict
                )
                )

    print('results:')
    print(json.dumps(results,indent=4))
    print('---')
    ##################################################
    # write to file
    ##################################################
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    save_file = os.path.join(results_path,
            '_'.join([dataset_name, est_name, str(random_state)])
            )

    print('save_file:',save_file)

    with open(save_file + '.json', 'w') as out:
        json.dump(jsonify(results), out, indent=4)

    return save_file + '.json'

################################################################################
# main entry point
################################################################################
import argparse
import importlib

if __name__ == '__main__':

    # parse command line arguments
    parser = argparse.ArgumentParser(
        description="Evaluate a method on a dataset.", add_help=False)
    parser.add_argument('INPUT_FILE', type=str,
                        help='Data file to analyze; ensure that the '
                        'target/label column is labeled as "class".')    
    parser.add_argument('-h', '--help', action='help',
                        help='Show this help message and exit.')
    parser.add_argument('-ml', action='store', dest='ALG',default=None,type=str, 
            help='Name of estimator (with matching file in methods/)')
    parser.add_argument('-results_path', action='store', dest='RDIR',
                        default='results_test', type=str, 
                        help='Name of save file')
    parser.add_argument('-seed', action='store', dest='RANDOM_STATE',
                        default=42, type=int, help='Seed / trial')
    parser.add_argument('-test',action='store_true', dest='TEST', 
                       help='Used for testing a minimal version')
    parser.add_argument('-stage',action='store', type=int, default=0,  
                       help='Competition stage')
    parser.add_argument('-n_jobs',action='store',  type=str, default='4',
                        help='number of cores available')
    parser.add_argument('-max_samples',action='store',  type=int, default=0,
                        help='number of training samples')

    args = parser.parse_args()
    set_env_vars(args.n_jobs)
    # import algorithm 
    print('import from','methods.'+args.ALG+'.regressor')
    algorithm = importlib.__import__('methods.'+args.ALG+'.regressor',
                                     globals(),
                                     locals(),
                                     ['*']
                                    )

    print('algorithm:',algorithm.est)

    # optional keyword arguments passed to evaluate
    eval_kwargs, test_params = {},{}
    if 'eval_kwargs' in dir(algorithm):
        eval_kwargs = algorithm.eval_kwargs

    if args.max_samples != 0:
        eval_kwargs['max_train_samples'] = args.max_samples

    evaluate_model(args.INPUT_FILE,
                   args.RDIR,
                   args.RANDOM_STATE,
                   args.ALG,
                   algorithm.est,  
                   algorithm.model, 
                   test = args.TEST, 
                   stage=args.stage,
                   **eval_kwargs
                  )
