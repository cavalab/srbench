from sklearn.base import BaseEstimator
import os
import subprocess
import pandas as pd
import numpy as np
this_dir = os.path.dirname(os.path.realpath(__file__))

class MRGPRegressor(BaseEstimator):
  def __init__(self, g=10, popsize=100, rt_mut=0.5, 
               rt_cross=0.5, max_len=10, n_jobs=1):
    env = dict(os.environ)
    self.g = g
    self.popsize = popsize
    self.rt_cross = rt_cross
    self.rt_mut = rt_mut
    self.max_len = max_len

  def fit(self, features, target, sample_weight=None, groups=None):
    data=pd.DataFrame(features)
    data['target']=target
    dataset = this_dir + '/tmp_data'
    data.to_csv(dataset+'-train',header=None, index=None)
    subprocess.check_output(['java', '-jar', 
                             this_dir+'/mrgp.jar',
                             '-train', 
                             dataset, 
                             str(self.g), 
                             str(self.popsize), 
                             str(self.rt_mut), 
                             str(self.rt_cross), 
                             str(self.max_len)
                            ])
    # get complexity
    df = pd.read_csv(dataset.replace('train','best'),header=None)
    self.model_ = df[2]
    print('model_:',self.model_)
    # count up parameters in model to get complexity
    self.complexity = df[2].apply(lambda x: str(x).count(' ')+1).values[0]
    print('complexity:',self.complexity)

    print('deleting tmp file',dataset)
    os.remove(dataset+'-train')
    return self

  def predict(self, test,ic=None):
    data=pd.DataFrame(test)
    data['tmp']=0
    dataset = this_dir + '/tmp_data'
    data.to_csv(dataset+'-test',header=None, index=None)

    y_pred=[float(x) for x in ''.join(
        chr(i) for i in 
        subprocess.check_output(['java', '-jar', 
                                 this_dir +'/mrgp.jar', '-test', 
                                 dataset])
        )[:-1].strip().split(" ")]

    if (len(y_pred)!=len(test) ):
      print("ERROR!")
    if (np.any(np.isinf(y_pred)) ):
      print("FOUND INFS!")
    if (np.any(np.isnan(y_pred)) ):
      print("FOUND NANs!")
      
    print('deleting tmp file',dataset)
    os.remove(dataset+'-test')
    return y_pred




