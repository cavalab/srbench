import sys
import itertools
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  
import warnings
import time
from tempfile import mkdtemp
from shutil import rmtree
from sklearn.externals.joblib import Memory
from read_file import read_file
from utils import feature_importance , roc
from convergence import convergence
import pdb
import numpy as np
import methods

def evaluate_model(dataset, save_file, random_state, est, hyper_params, 
                  complexity):

    est_name = type(est).__name__
    
    # load data
    features, labels, feature_names = read_file(dataset)
    # generate train/test split
    X_train, X_test, y_train, y_test = train_test_split(features, labels,
                                                        train_size=0.75,
                                                        test_size=0.25,
                                                        random_state=None)
    # scale and normalize the data
    X_train = StandardScaler().fit_transform(X_train)
    sc_y = StandardScaler()
    y_train = sc_y.fit_transform(y_train.reshape(-1,1)).flatten()
    print('X_train:',X_train.shape)
    print('y_train:',y_train.shape)
    # define CV strategy for hyperparam tuning
    cv = KFold(n_splits=5, shuffle=True,random_state=random_state)
    grid_est = GridSearchCV(est,cv=cv, param_grid=hyper_params,
            verbose=1,n_jobs=-1,scoring='r2',error_score=0.0)

    t0 = time.process_time()
    # Grid Search
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid_est.fit(X_train,y_train)

    runtime = time.process_time() - t0

    best_est = grid_est.best_estimator_
    
    # get the size of the final model
    if complexity == None:
        model_size = features.shape[1]
    else:
        model_size = complexity(best_est)

    # scores
    sc_inv = sc_y.inverse_transform
    pred = grid_est.predict
    # mse
    train_score_mse = mean_squared_error(sc_inv(y_train), 
                                         sc_inv(pred(X_train)))
    test_score_mse = mean_squared_error(y_test, sc_inv(pred(X_test)))
    # mae 
    train_score_mae = mean_absolute_error(sc_inv(y_train), 
                                          sc_inv(pred(X_train)))
    test_score_mae = mean_absolute_error(y_test, sc_inv(pred(X_test)))
    # r2 
    train_score_r2 = r2_score(sc_inv(y_train), sc_inv(pred(X_train)))
    test_score_r2 = r2_score(y_test, sc_inv(pred(X_test)))

    sorted_grid_params = ','.join(['{}={}'.format(p, v) 
                             for p,v in sorted(best_est.get_params().items())])


    # print results
    out_text = '\t'.join([dataset.split('/')[-1][:-7],
                          est_name,
                          str(sorted_grid_params).replace('\n',','),
                          str(random_state),
                          str(train_score_mse),
                          str(train_score_mae),
                          str(train_score_r2),
                          str(test_score_mse),
                          str(test_score_mae),
                          str(test_score_r2),
                          str(runtime),
                          str(model_size)
                          ]
                          )

    print(out_text)
    sys.stdout.flush()
    if save_file:
        with open(save_file, 'a') as out:
            out.write(out_text+'\n')

        # store CV detailed results
        df = pd.DataFrame(data=grid_est.cv_results_)
        df['seed'] = random_state
        cv_save_name = save_file.split('.csv')[0]+'_cv_results.csv'
        import os.path
        if os.path.isfile(cv_save_name):
            # if exists, append
            df.to_csv(cv_save_name, mode='a', header=False, index=False)
        else:
            df.to_csv(cv_save_name, index=False)

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
    parser.add_argument('-save_file', action='store', dest='SAVE_FILE',
                        default=None, type=str, help='Name of save file')
    parser.add_argument('-seed', action='store', dest='RANDOM_STATE',
                        default=None, type=int, help='Seed / trial')

    args = parser.parse_args()
    # import algorithm 
    print('import from','methods.'+args.ALG)
    algorithm = importlib.__import__('methods.'+args.ALG,globals(),locals(),
                                   ['est','hyper_params','complexity'])
    if args.ALG == 'mrgp':
        algorithm.est.dataset=args.INPUT_FILE.split('/')[-1][:-7]

    print('algorithm:',algorithm.est)
    print('hyperparams:',algorithm.hyper_params)
    evaluate_model(args.INPUT_FILE, args.SAVE_FILE, args.RANDOM_STATE, 
                   algorithm.est, algorithm.hyper_params)
