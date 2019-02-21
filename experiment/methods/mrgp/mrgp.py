from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer, Imputer

import os
import subprocess
import pandas as pd
import numpy as np


class MRGPClassifier(BaseEstimator):
  def __init__(self, dataset, g=10, popsize=100, rt_mut=0.5, rt_cross=0.5, max_len=10, n_jobs=1):
    env = dict(os.environ)
    self.dataset = dataset
    self.g = g
    self.popsize = popsize
    self.rt_cross = rt_cross
    self.rt_mut = rt_mut
    self.max_len = max_len

  def fit(self, features, target, sample_weight=None, groups=None):
    data=pd.DataFrame(features)
    data['target']=target
    data.to_csv(self.dataset+"-train",header=None, index=None)
    subprocess.check_output(['java', '-jar', '/net/archive/groups/plggbicl/ellyn/methods/mrgp/mrgp.jar', '-train', self.dataset, str(self.g), str(self.popsize), str(self.rt_mut), str(self.rt_cross), str(self.max_len)])
    return self;

  def predict(self, test,ic=None):
    data=pd.DataFrame(test)
    data['tmp']=0
    data.to_csv(self.dataset+"-test",header=None, index=None)

    y_pred=[float(x) for x in ''.join(chr(i) for i in subprocess.check_output(['java', '-jar', '/net/archive/groups/plggbicl/ellyn/methods/mrgp/mrgp.jar', '-test', self.dataset]))[:-1].strip().split(" ")]
    if (len(y_pred)!=len(test) ):
      print("ERROR!")
    if (np.any(np.isinf(y_pred)) ):
      print("FOUND INFS!")
    if (np.any(np.isnan(y_pred)) ):
      print("FOUND NANs!")
      
    return y_pred




