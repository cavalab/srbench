from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline, make_union

import os
import subprocess
import pandas as pd
import numpy as np
import time

class GSGPClassifier(BaseEstimator):




#  def __init__(self, dataset, y_train, y_test, g=10, popsize=100, rt_mut=0.5, rt_cross=0.5, max_len=10, n_jobs=1):
#  def __init__(self, dataset, y_train, y_test, g=10, popsize=100, rt_mut=0.5, rt_cross=0.5, max_len=10, n_jobs=1):
  def __init__(self, dataset, y_train, y_test, g=100, popsize=1000, rt_mut=0.5, rt_cross=0.5, max_len=10, n_jobs=1):
    env = dict(os.environ)
    #env['JAVA_OPTS'] = 'foo'
    self.dataset = dataset
    self.g = g
    self.popsize = popsize
    self.rt_cross = rt_cross
    self.rt_mut = rt_mut
    self.max_len = max_len
    self.trainsize=-1
    self.nvar=-1
    self.X_train=None
    self.y_train=y_train
    self.X_test=None
    self.y_test=y_test



  def line_prepender(self,filename, line):
    with open(filename, 'r+') as f:
      content = f.read()
      f.seek(0, 0)
      f.write(line.rstrip('\r\n') + '\n' + content)


  def fit(self, X_train, y_train, sample_weight=None):
    self.X_train=X_train
    self.y_train=y_train
    self.y_test=y_train
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
'''.format(self.popsize, self.g, self.rt_cross,self.rt_mut,self.max_len)
    ffile=open(self.dataset+"-configuration.ini","w")
    ffile.write(text)


  def predict(self, X_test,ic=None):

    # train data
    data=pd.DataFrame(self.X_train)
    data['target']=self.y_train
    data.to_csv(self.dataset+"-training",header=None, index=None, sep='\t')
    trainsize=self.X_train.shape[0]
    nvar=self.X_train.shape[1]
    self.line_prepender(self.dataset+'-training',str(trainsize)+'\n')
    self.line_prepender(self.dataset+'-training',str(nvar)+'\n')
    time.sleep(1)
    # test data 1
    datat1=pd.DataFrame(X_test)
    datat1['target']=self.y_test[0:len(X_test)]
    datat1.to_csv(self.dataset+"-test1",header=None, index=None, sep='\t')
    testsize=X_test.shape[0]
    nvar=X_test.shape[1]
    time.sleep(1)
    self.line_prepender(self.dataset+'-test1',str(X_test.shape[0])+'\n')
    self.line_prepender(self.dataset+'-test1',str(X_test.shape[1])+'\n')
    time.sleep(1)
    #test data 2
    datat2 = datat1.drop('target',axis=1)
    datat2.to_csv(self.dataset+"-test",header=None, index=None, sep='\t')
    self.line_prepender(self.dataset+'-test',str(X_test.shape[1])+'\n')


    #do training
    subprocess.call(["sed -i -e 's/USE_TEST_SET.*/USE_TEST_SET = 0/g' "+self.dataset+"-configuration.ini"],shell=True)
    subprocess.call(["sed -i -e 's/expression_file.*/expression_file = 0/g' "+self.dataset+"-configuration.ini"],shell=True)
    #print('./gsgp_original -train_file '+self.dataset+'-training -test_file ' + self.dataset+'-test1' + " -name "+ self.dataset )
    subprocess.call(['./gsgp_original -train_file '+self.dataset+'-training -test_file ' + self.dataset+'-test1'+" -name "+ self.dataset],shell=True)
    time.sleep(1)
    #do testing
    #print("sed -i -e ''s/USE_TEST_SET.*/USE_TEST_SET = 1/g'' "+self.dataset+"-configuration.ini")
    subprocess.call(["sed -i -e 's/USE_TEST_SET.*/USE_TEST_SET = 1/g' "+self.dataset+"-configuration.ini"],shell=True)
    subprocess.call(["sed -i -e 's/expression_file.*/expression_file = 1/g' "+self.dataset+"-configuration.ini"],shell=True)
    #print('./gsgp_original -test_file '+self.dataset+'-test'+" -name "+ self.dataset)
    subprocess.call(['./gsgp_original -test_file '+self.dataset+'-test'+" -name "+ self.dataset],shell=True)
    time.sleep(1)
    y_pred2=[]
    with open(self.dataset+'-evaluation_on_unseen_data.txt','r') as f:
      for line in f:
        y_pred2.append(float(line.strip()))
    y_pred2=y_pred2[:-1]
    return y_pred2
