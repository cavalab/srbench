import sys
import itertools
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.pipeline import Pipeline,make_pipeline
from metrics import balanced_accuracy_score
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

def evaluate_model(dataset, save_file, random_state, est, hyper_params):

    est_name = type(est).__name__
    
    # load data
    features, labels, feature_names = read_file(dataset)
    X = StandardScaler().fit_transform(input_data.drop(TARGET_NAME, axis=1).values.astype(float))

    sc_y = StandardScaler()
    y = sc_y.fit_transform(input_data[TARGET_NAME].values.reshape(-1,1))
    # generate train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        train_size=0.75,
                                                        test_size=0.25,
                                                        random_state=None)
    # define CV strategy for hyperparam tuning
    cv = KFold(n_splits=5, shuffle=True,random_state=random_state)
    grid_est = GridSearchCV(est,cv=cv, param_grid=hyper_params,
            verbose=1,n_jobs=1,scoring='r2',error_score=0.0)
    
    with warnings.catch_warnings():
        # Squash warning messages. Turn this off when debugging!
        # warnings.simplefilter('ignore')
        
        t0 = time.process_time()
        # generate cross-validated predictions for each data point 
        # using the best estimator 
        grid_est.fit(X_train,y_train)
        
        runtime = time.process_time() - t0

        best_est = grid_est.best_estimator_
        
        # get the size of the final model
        model_size= complexity(best_est)

        param_string = ','.join(['{}={}'.format(p, v) for p,v in 
                                 best_est.get_params().items()])

        # scores
        sc_inv = sc_y.inverse_transform
        pred = grid_est.predict

        train_score_mse = mean_squared_error(sc_inv(y_train),
                                             sc_inv(pred(X_train)))
        train_score_mae = mean_absolute_error(sc_inv(y_train),
                                              sc_inv(pred(X_train)))
        test_score_mse = mean_squared_error(sc_inv(y_test),
                                            sc_inv(pred(X_test)))
        test_score_mae = mean_absolute_error(sc_inv(y_test),
                                             sc_inv(pred(X_test)))

        sorted_grid_params = sorted(grid_est.best_params_.items(), 
                                    key=operator.itemgetter(0))

        best=grid_est.best_estimator_
        best.stack_2_eqn(best.best_estimator_)


        # print results
        out_text = '\t'.join([dataset.split('/')[-1][:-7],
                              est_name,
                              str(random_state),
                              str(sorted_grid_params).replace('\n',','),
                              str(train_score_mse),
                              str(train_score_mae),
                              str(test_score_mse),
                              str(test_score_mae),
                              str(runtime),
                              str(model_size)
                              # TODO: add complexity 
                              ]
                              )

        print(out_text)
        sys.stdout.flush()
        with open(save_file, 'a') as out:
            out.write(out_text+'\n')
        sys.stdout.flush()


        df = pd.DataFrame(data=grid_est.cv_results_)
        df['seed'] = random_state
        cv_save_name = save_file.split('.csv')[0]+'_cv_results.csv'
        import os.path
        if os.path.isfile(cv_save_name):
            # if exists, append
            df.to_csv(cv_save_name, mode='a', header=False, index=False)
        else:
            df.to_csv(cv_save_name, index=False)
