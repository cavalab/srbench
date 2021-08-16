from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline, make_union

import os
import subprocess
import pandas as pd
import numpy as np
import time

this_dir = os.path.dirname(os.path.realpath(__file__))


class GSGPRegressor(BaseEstimator):  

  def __init__(self, g=100, popsize=200, rt_mut=0.5, rt_cross=0.5,
              max_len=6, n_jobs=1, random_state=None):
    env = dict(os.environ)
    self.g = g
    self.popsize = popsize
    self.rt_cross = rt_cross
    self.rt_mut = rt_mut
    self.max_len = max_len
    self.trainsize=-1
    self.nvar=-1
    self.exe_name = 'GP'
    self.random_state=random_state

  def line_prepender(self,filename, line):
    with open(filename, 'r+') as f:
      content = f.read()
      f.seek(0, 0)
      f.write(line.rstrip('\r\n') + '\n' + content)

  def fit(self, X_train, y_train, sample_weight=None):
    # Define tmp file names
    self.dataset = this_dir + '/tmp_data_' + str(np.random.randint(2**15-1))
    self.dataset_short = self.dataset.split('/')[-1]

    # Config file
    text='''population_size={}
max_number_generations={}
init_type = 2
p_crossover={}
p_mutation={}
max_depth_creation={}
tournament_size= 4
zero_depth = 0
mutation_step = 1
num_random_constants = 0
min_random_constant = -100
max_random_constant = 100
minimization_problem = 1
random_tree = 500
expression_file = 0
USE_TEST_SET = 0
'''.format(self.popsize, self.g, self.rt_cross, self.rt_mut, self.max_len)
    ffile=open(self.dataset+"-configuration.ini","w")
    ffile.write(text)
    ffile.close()

    # train data
    data=pd.DataFrame(X_train, copy=True)
    data['target']=y_train
    train_file = self.dataset+"-train"
    data.to_csv(train_file,header=None, index=None, sep='\t')
    trainsize=X_train.shape[0]
    nvar=X_train.shape[1]
    self.line_prepender(train_file,str(trainsize)+'\n')
    self.line_prepender(train_file,str(nvar)+'\n')
    del data
    # empty test data
    empty_test_data = self.dataset+'-test0'
    subprocess.call(['touch', empty_test_data])
    time.sleep(1)
    #do training
    subprocess.call(["sed -i -e 's/USE_TEST_SET.*/USE_TEST_SET = 0/g' "
                     +self.dataset+"-configuration.ini"],shell=True)
    subprocess.call(["sed -i -e 's/expression_file.*/expression_file = 0/g' "
                     +self.dataset+"-configuration.ini"],shell=True)
    subprocess.call(' '.join([this_dir+'/'+self.exe_name,
                     '-train_file '+train_file,
                     '-test_file', empty_test_data,
                     ' -name '+self.dataset,
                     (self.random_state != None)*('-seed '+str(self.random_state))]),
                     shell=True)
    time.sleep(1)
    os.remove(self.dataset+"-train")
    os.remove(self.dataset+"-test0")


  def predict(self, X_test,ic=None):
    # test data
    if not isinstance(X_test, pd.DataFrame):
      X_test = pd.DataFrame(X_test)
    X_test.to_csv(self.dataset+"-test",header=None, index=None, sep='\t')
    self.line_prepender(self.dataset+'-test',str(X_test.shape[1])+'\n') 
    
    #do testing
    subprocess.call(["sed -i -e 's/USE_TEST_SET.*/USE_TEST_SET = 1/g' "
                     +self.dataset+"-configuration.ini"],shell=True)
    subprocess.call(["sed -i -e 's/expression_file.*/expression_file = 1/g' "
                     +self.dataset+"-configuration.ini"],shell=True)
    subprocess.call(' '.join([this_dir+'/'+self.exe_name,
                     '-test_file', self.dataset+'-test',
                     '-name', self.dataset]),
                     shell=True)    
    time.sleep(1)
    y_pred=[]
    with open(self.dataset+'-evaluation_on_unseen_data.txt','r') as f:
      for line in f:
        y_pred.append(float(line.strip()))
    y_pred=y_pred[:-1]
    assert(len(y_pred) == len(X_test))
    os.remove(self.dataset+"-test")
    return y_pred
