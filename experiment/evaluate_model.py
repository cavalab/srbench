import sys
import itertools
import pandas as pd
from sklearn.experimental import enable_halving_search_cv # noqa
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
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

def evaluate_model(dataset, results_path, random_state, est_name, est, 
                   hyper_params, complexity, model, n_samples=10000, 
                   test=False):

    print(40*'=','Evaluating '+est_name+' on ',dataset,40*'=',sep='\n')
    if hasattr(est, 'random_state'):
        est.random_state = random_state

    ##################################################
    # setup data
    ##################################################
    features, labels, feature_names = read_file(dataset)
    # if dataset is large, subsample it 
    if n_samples > 0 and len(labels) > n_samples:
        print('subsampling data from',len(labels),'to',n_samples)
        sample_idx = np.random.choice(np.arange(labels), size=n_samples)
        labels = labels[sample_idx]
        features = features[sample_idx]


    # generate train/test split
    X_train, X_test, y_train, y_test = train_test_split(features, labels,
                                                    train_size=0.75,
                                                    test_size=0.25,
                                                    random_state=random_state)
    # scale and normalize the data
    sc_X, sc_y = StandardScaler(), StandardScaler()

    X_train_scaled = sc_X.fit_transform(X_train)
    X_test_scaled = sc_X.transform(X_test)

    y_train_scaled = sc_y.fit_transform(y_train.reshape(-1,1)).flatten()
    y_test_scaled = sc_y.transform(y_test.reshape(-1,1)).flatten()

    print('X_train:',X_train_scaled.shape)
    print('y_train:',y_train_scaled.shape)

    ################################################## 
    # define CV strategy for hyperparam tuning
    ################################################## 
    # define a test mode with fewer splits, no hyper_params, and few iterations
    if test:
        print('test mode enabled')
        n_splits = 2
        hyper_params = {}
        print('hyper_params set to',hyper_params)
        for genname in ['generations','gens','g','itrNum','treeNum']:
            if hasattr(est, genname):
                print('setting',genname,'=2 for test')
                setattr(est, genname, 2)
        if hasattr(est, 'popsize'):
            print('setting popsize=5 for test')
            est.popsize = 20 
        if hasattr(est, 'val'):
            print('setting val=1 for test')
            est.val = 1
    else:
        n_splits = 5

    cv = KFold(n_splits=n_splits, shuffle=True,random_state=random_state)

    grid_est = HalvingGridSearchCV(est,cv=cv, param_grid=hyper_params,
            verbose=2,n_jobs=1,scoring='r2',error_score=0.0)

    ################################################## 
    # Fit models
    ################################################## 
    t0 = time.process_time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid_est.fit(X_train_scaled, y_train_scaled)
    runtime = time.process_time() - t0
    print('Training took',runtime,'seconds')
    best_est = grid_est.best_estimator_
    # best_est = grid_est
    
    ##################################################
    # store results
    ##################################################
    dataset_name = dataset.split('/')[-1][:-7]
    results = {
        'dataset':dataset_name,
        'algorithm':est_name,
        'params':{k:v for k,v in best_est.get_params().items() 
                  if any(isinstance(v, t) for t in [bool,int,float,str])},
        'random_state':random_state,
        'runtime':runtime 
    }

    # get the size of the final model
    if complexity == None:
        results['model_size'] = int(features.shape[1])
    else:
        results['model_size'] = int(complexity(best_est))

    # get the final symbolic model as a string
    if model == None:
        results['symbolic_model'] = 'not implemented'
    else:
        if 'X' in inspect.signature(model).parameters.keys():
            results['symbolic_model'] = model(best_est, X_train_scaled)
        else:
            results['symbolic_model'] = model(best_est)

    # scores
    sc_inv = sc_y.inverse_transform
    pred = grid_est.predict

    for fold, target, X in zip(['train','test'],
                               [y_train, y_test], 
                               [X_train, X_test]
                              ):
        for score, scorer in [('mse',mean_squared_error),
                              ('mae',mean_absolute_error),
                              ('r2', r2_score)
                             ]:
            results[score + '_' + fold] = scorer(target, sc_inv(pred(X))) 
    # # mse
    # results['train_score_mse'] = mean_squared_error(sc_inv(y_train), 
    #                                                 sc_inv(pred(X_train)))
    # results['test_score_mse'] = mean_squared_error(y_test, 
    #                                                sc_inv(pred(X_test)))

    # # mae 
    # results['train_score_mae'] = mean_absolute_error(sc_inv(y_train), 
    #                                                  sc_inv(pred(X_train)))
    # results['test_score_mae'] = mean_absolute_error(y_test, 
    #                                                 sc_inv(pred(X_test)))

    # # r2 
    # results['train_score_r2'] = r2_score(sc_inv(y_train), 
    #                                      sc_inv(pred(X_train)))
    # results['test_score_r2'] = r2_score(y_test, sc_inv(pred(X_test)))

    
    ##################################################
    # write to file
    ##################################################
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    save_file = (results_path + '/' + dataset_name + '_' + est_name + '_' 
                 + str(random_state))
    print('save_file:',save_file)

    print('results types:',type(results))
    for k,v in results.items():
        print(k,v.__class__.__name__ )
    with open(save_file + '.json', 'w') as out:
        json.dump(results, out)

    # store CV detailed results
    cv_results = grid_est.cv_results_
    cv_results['random_state'] = random_state
    for k,v in cv_results.items():
        print(k,'type:',type(v).__name__)
        if type(v).__name__ in ['ndarray','MaskedArray']:
            cv_results[k] = cv_results[k].tolist()
    with open(save_file + '_cv_results.json', 'w') as out:
        json.dump(cv_results, out)

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

    args = parser.parse_args()
    # import algorithm 
    print('import from','methods.'+args.ALG)
    algorithm = importlib.__import__('methods.'+args.ALG,
                                     globals(),
                                     locals(),
                                     ['est',
                                      'hyper_params',
                                      'complexity',
                                      'model'
                                     ]
                                    )
    if args.ALG == 'mrgp':
        algorithm.est.dataset=args.INPUT_FILE.split('/')[-1][:-7]

    print('algorithm:',algorithm.est)
    print('hyperparams:',algorithm.hyper_params)
    evaluate_model(args.INPUT_FILE, args.RDIR, args.RANDOM_STATE, args.ALG,
                   algorithm.est, algorithm.hyper_params, algorithm.complexity,
                   algorithm.model, args.TEST)
