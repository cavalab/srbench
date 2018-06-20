import sys
import time

import operator
import pandas as pd
import numpy as np
import itertools
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
#from sklearn.svm import SVC
#from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.metrics import *
import metrics
from sklearn.model_selection import train_test_split
#from sklearn.pipeline import make_pipeline
import itertools
#import pdb

from gsgp import GSGPClassifier

# read arguments (1: dataset name, 2: output file name, 3: trial #, 4: # cores)
dataset = sys.argv[1]
#output_file = sys.argv[2]
trial = sys.argv[3]

# Read the data set into memory
input_data = pd.read_csv(dataset, compression='gzip', sep='\t')

#header
#with open(output_file,'w') as out:
#    out.write('dataset\tmethod\ttrial\tparameters\taccuracy\ttime\n')
#print('dataset\tmethod\ttrial\tparameters\taccuracy\ttime')

TARGET_NAME = 'target'
INPUT_SEPARATOR = '\t'
n_jobs = 1






hyper_params = [
    {
        'popsize': (100,), 'g': (1000,),
        'max_len': (6,),
        'rt_cross':(0.0,),'rt_mut':(1.0,),
    },
    {
        'popsize': (100,), 'g': (1000,),
        'max_len': (6,),
        'rt_cross':(0.1,),'rt_mut':(0.9,),
    },
    {
        'popsize': (100,), 'g': (1000,),
        'max_len': (6,),
        'rt_cross':(0.2,),'rt_mut':(0.8,),
    },
    {
        'popsize': (1000,), 'g': (100,),
        'max_len': (6,),
        'rt_cross':(0.0,),'rt_mut':(1.0,),
    },
    {
        'popsize': (1000,), 'g': (100,),
        'max_len': (6,),
        'rt_cross':(0.1,),'rt_mut':(0.9,),
    },
    {
        'popsize': (1000,), 'g': (100,),
        'max_len': (6,),
        'rt_cross':(0.2,),'rt_mut':(0.8,),
    },
]










sc_y = StandardScaler()
X = StandardScaler().fit_transform(input_data.drop(TARGET_NAME, axis=1).values.astype(float))
y = sc_y.fit_transform(input_data[TARGET_NAME].values.reshape(-1,1))





X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=0.75,
                                                    test_size=0.25,
                                                    random_state=None)


est=GSGPClassifier(dataset=dataset.split('/')[-1][:-7], y_test=y_test, y_train=y_test)



grid_clf = GridSearchCV(est,cv=5,param_grid=hyper_params,
                   verbose=0,n_jobs=n_jobs,scoring='r2')

#fit model
#get fit time
t0 = time.time()
grid_clf.fit(X_train,y_train)
runtime = time.time()-t0




train_score_mse = mean_squared_error(sc_y.inverse_transform(y_train),sc_y.inverse_transform(grid_clf.predict(X_train)))
train_score_mae = mean_absolute_error(sc_y.inverse_transform(y_train),sc_y.inverse_transform(grid_clf.predict(X_train)))
test_score_mse = mean_squared_error(sc_y.inverse_transform(y_test),sc_y.inverse_transform(grid_clf.predict(X_test)))
test_score_mae = mean_absolute_error(sc_y.inverse_transform(y_test),sc_y.inverse_transform(grid_clf.predict(X_test)))


sorted_grid_params = sorted(grid_clf.best_params_.items(), key=operator.itemgetter(0))


# print results
out_text = '\t'.join([dataset.split('/')[-1][:-7],
                      'gsgp',
                      str(trial),
                      str(sorted_grid_params).replace('\n',','),
                      str(train_score_mse),
                      str(train_score_mae),
                      str(test_score_mse),
                      str(test_score_mae),
                      str(runtime)])


print(out_text)
sys.stdout.flush()

