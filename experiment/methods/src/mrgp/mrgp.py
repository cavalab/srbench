import itertools
from sklearn.base import BaseEstimator
import os
import subprocess
import pandas as pd
import numpy as np
from pandas.util import hash_pandas_object
import hashlib
THIS_DIR = os.path.dirname(os.path.realpath(__file__))

class MRGPRegressor(BaseEstimator):
  def __init__(self, g=10, popsize=100, rt_mut=0.5, 
               rt_cross=0.5, max_len=10, time_out=10*60,
               tmp_dir=None
               ):
    self.g = g
    self.popsize = popsize
    self.rt_cross = rt_cross
    self.rt_mut = rt_mut
    self.max_len = max_len
    self.time_out = time_out  #in seconds
    self.tmp_dir = tmp_dir

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
    subprocess.check_output(['java', '-jar', 
                             THIS_DIR+'/mrgp.jar',
                             '-train', 
                             self.dataset, 
                             str(self.g), 
                             str(self.popsize), 
                             str(self.rt_mut), 
                             str(self.rt_cross), 
                             str(self.max_len),
                             str(self.time_out)
                            ])
    # get complexity
    # print('reading in ', self.dataset+'best')
    df = pd.read_csv(self.dataset+'-best',header=None)
    # print(self.dataset+'-best:',df)
    self.model_ = open(self.dataset+'-best','r').readline()
    # self.model_ = df[4]
    # print('model_:',self.model_)
    # count up model components to get complexity
    self.complexity = len(list(itertools.chain(
        *[m.split(' ') for m in self.model_.split(',')])))
    # print('complexity:',self.complexity)

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




