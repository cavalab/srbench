import itertools
from sklearn.base import BaseEstimator
import os
import re
import subprocess
import pandas as pd
import numpy as np
from pandas.util import hash_pandas_object
import hashlib
THIS_DIR = os.path.dirname(os.path.realpath(__file__))

class MRGPRegressor(BaseEstimator):
  def __init__(self, g=10, popsize=100, rt_mut=0.5, 
               rt_cross=0.5, max_len=10, time_out=10*60,
               tmp_dir=None, n_jobs=1, random_state=None
               ):
    self.g = g
    self.popsize = popsize
    self.rt_cross = rt_cross
    self.rt_mut = rt_mut
    self.max_len = max_len
    self.time_out = time_out  #in seconds
    self.tmp_dir = tmp_dir
    self.n_jobs = n_jobs
    self.random_state = random_state

  def fit(self, features, target, sample_weight=None, groups=None):
    data=pd.DataFrame(features)
    data['target']=target
    rowHashes = hash_pandas_object(data).values
    filehash = hashlib.sha256(rowHashes).hexdigest()
    if self.tmp_dir != None:
        data_dir = self.tmp_dir
    else:
        data_dir = THIS_DIR

    self.dataset = (data_dir + '/tmp_data_' 
                    + filehash + '_' 
                    + str(np.random.randint(2**15-1))
                    )
    # print('dataset name:',self.dataset)
    data.to_csv(self.dataset+'-train',
                header=None, 
                index=None)
    output = ['java', '-jar', THIS_DIR+'/mrgp.jar',
             '-train',
             self.dataset,
             str(self.g),
             str(self.popsize),
             str(self.rt_mut),
             str(self.rt_cross),
             str(self.max_len),
             str(self.time_out),
             str(self.n_jobs)
             ]
    output = output+[str(self.random_state)] if self.random_state != None else output
    subprocess.check_output(output)
    # get model and complexity
    self.model_, self.complexity_ = self._get_model()

    # print('deleting training file',self.dataset+'-train')
    os.remove(self.dataset+'-train')

    return self

  def predict(self, test,ic=None):
    data=pd.DataFrame(test)
    data['tmp']=0
    data.to_csv(self.dataset+'-test',header=None, index=None)

    y_pred=[float(x) for x in ''.join(
        chr(i) for i in 
        subprocess.check_output(['java', '-jar', 
                                 THIS_DIR +'/mrgp.jar', '-test', 
                                 self.dataset])
        )[:-1].strip().split(" ")]

    if (len(y_pred)!=len(test) ):
      print("ERROR!")
    if (np.any(np.isinf(y_pred)) ):
      print("FOUND INFS!")
    if (np.any(np.isnan(y_pred)) ):
      print("FOUND NANs!")
      
    # print('deleting tmp file',self.dataset)
    os.remove(self.dataset+'-test')
    return y_pred

  def _get_model(self):
    """reads in best model and gets a string version with complexity"""
    best_data = open(self.dataset+'-best','r').readline().split(',')
    # file structure:
    # 0 mintarget, 1 maxtarget, 2 weights, 3 intercept, 4 model_str
    internal_weights=best_data[2]
    intercept = best_data[3]
    model_form = best_data[4]
    # rename functions to python operator names
    model_form = model_form.replace('mydivide','div')
    model_form = model_form.replace('*','mul')
    model_form = model_form.replace('-','sub')
    model_form = model_form.replace('+','add')
    # move starting paren to other side of functions
    model_form = re.sub(
                    pattern=r'\((.+?(?= ))',
                    repl=r'\1(',
                    string=model_form
                   )
    complexity_ = 2+len(internal_weights)*3
    model_ = ' '.join([b+'*'+ m for b,m in zip(internal_weights.split(' '),
                                      model_form.split(' '))])

    return model_, complexity_
