import numpy as np
from sklearn.model_selection import train_test_split
import time
import pandas as pd
import pdb

def convergence(est, est_name, X, y, save_file, random_state):
    """Stores logging information about validation scores vs time for estimators."""
    if 'Feat' in est_name:
        feat_convergence(est, est_name, X, y, save_file, random_state)
    elif 'XGB' in est_name:
        xgb_convergence(est, est_name, X, y, save_file, random_state)
    
def feat_convergence(est, est_name, X, y, save_file, random_state):
    """Add logfile to estimator and refit to data."""
    # FEAT internally will split the data .75/.25 and report validation on the .25. 
    # Turn on logging for the FEAT estimator and fit to data

    est.logfile = (save_file.split('.csv')[0] + '_' + str(random_state) + '.log').encode()
    est.fit(X,y)

def xgb_convergence(est, est_name, X, y, save_file, random_state):
    #XGBoost provides logging options that we have to set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
            random_state=random_state)
    # time estimate
    t0 = time.process_time()

    est.fit(X_train,y_train, 
            eval_set = [(X_train,y_train),(X_test,y_test)], 
            eval_metric = 'rmse')
    runtime = time.process_time() - t0
    
    train_loss = np.square(est.evals_result_['validation_0']['rmse'])
    val_loss = np.square(est.evals_result_['validation_1']['rmse'])
    runtimes = np.linspace(0,runtime,len(train_loss))
   
    df = pd.DataFrame(data = {'time':runtimes,
                              'train_loss':train_loss,
                              'val_loss':val_loss})
    log_file = save_file.split('.csv')[0] + '_' + str(random_state) + '.log'
    df.to_csv(log_file, index=None)
