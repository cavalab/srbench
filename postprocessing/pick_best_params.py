#!/usr/bin/env python
# coding: utf-8

# # saved tuned models
# 
# Go thru PMLB results and pick the most common params in the best estimators. 
# Use these parameter settings for the Feynman and Strogatz models.

import pandas as pd
import json
import numpy as np
import os

symbolic_algs = [
    'AFP',
    'AFP_FE',
    'AIFeynman',
    'BSR',
    'DSR',
    'FFX',
    'FEAT',
    'EPLEX',
    'GP-GOMEA',
    'gplearn',
    'ITEA',
    'MRGP',
    'Operon',
    'SBP-GP',
]
sr_filenames = [
    'AFPRegressor',
    'FE_AFPRegressor',
    'AIFeynman',
    'BSRRegressor',
    'DSRRegressor',
    'FFXRegressor',
    'FEATRegressor',
    'EPLEXRegressor',
    'GPGOMEARegressor',
    'gplearn',
    'ITEARegressor',
    'MRGPRegressor',
    'OperonRegressor',
    'sembackpropgp'
    ]
sr_name_to_filename = {k:v for k,v in zip(symbolic_algs, sr_filenames)}
# read data from feather
df_results = pd.read_feather('../results/black-box_results.feather')

# keep only symbolic regressors
df_results = df_results.loc[df_results.algorithm.isin(symbolic_algs)]
# turn params in to string
# df_results['params_str'] = df_results['params'].apply(str)

# save so we don't have to load again
# df_results[['algorithm','params_str']].to_feather(rdir.replace('.','').replace('/','')+'_params.feather')


# ## find the mode of each algorithm's params

# In[3]:


# df_results.params.apply(str)
best_params = []
for alg, dfg in df_results.groupby('algorithm'):
    counts = dfg['params_str'].value_counts()
    print(alg, 'mode of params:',counts.index[0])
    best_params.append([alg, counts.index[0]])

# best_params                    


# # write tuned model scripts for each algorithm

# In[5]:


import os
for alg, bp in best_params:
#     os.system('cp ../experiment/methods/{ALG}.py ../experiment/methods/tuned/{ALG}.py'.format(ALG=alg))
    param_file = sr_name_to_filename[alg]
    with open(f'../experiment/methods/tuned/params/_{param_file}.py','w') as f:
        f.write('params = {}\n'.format(bp))
