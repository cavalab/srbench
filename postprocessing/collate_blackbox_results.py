"""Collates json-formatted results, cleans them up and saves them as .feather
files."""
# Author: William La Cava, williamlacava@gmail.com
# SRBENCH
# License: GPLv3

################################################################################
# Black-box problems
################################################################################
import pandas as pd
import json
import numpy as np
from glob import glob
from tqdm import tqdm
import os
import sys

rdir = '../results_pmlb_r1/'

if len(sys.argv) > 1:
    rdir = sys.argv[1]
else:
    print('no rdir provided, using',rdir)
print('reading results from  directory', rdir)
    
symbolic_algs = [
    'AFP', 
    'AFP_FE',
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
    'AIFeynman'
]
nongp_algs = [
    'BSR',
    'DSR',
    'AIFeynman'
]
gp_algs = [
    'AFP', 
    'AFP_FE',
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
##########
# load data from json
##########
frames = []
comparison_cols = [
    'dataset',
    'algorithm',
    'random_state',
    'time_time',
    'model_size',
    'symbolic_model',
    'r2_test',
    'mse_test',
    'mae_test',
    'params'
]
fails = []
import pdb
for f in tqdm(glob(rdir + '/*/*.json')):
    if 'cv_results' in f: 
        continue
    # leave out symbolic data
    if 'feynman_' in f or 'strogatz_' in f:
        continue
    # leave out LinearReg, Lasso (we have SGD with penalty)
    if any([m in f for m in ['LinearRegression','Lasso','EHCRegressor']]):
        continue
    try: 
        r = json.load(open(f,'r'))
        if isinstance(r['symbolic_model'],list):
#             print(f)
            sm = ['B'+str(i)+'*'+ri for i, ri in enumerate(r['symbolic_model'])]
            sm = '+'.join(sm)
            r['symbolic_model'] = sm
            
        sub_r = {k:v for k,v in r.items() if k in comparison_cols}
    #     df = pd.DataFrame(sub_r)
        frames.append(sub_r) 
    #     print(f)
    #     print(r.keys())
    except Exception as e:
        fails.append([f,e])
        pass
    
print(len(fails),'fails:',fails)
# df_results = pd.concat(frames)
df_results = pd.DataFrame.from_records(frames)
df_results['params_str'] = df_results['params'].apply(str)
df_results = df_results.drop(columns=['params'])
##########
# cleanup
##########
df_results = df_results.rename(columns={'time_time':'training time (s)'})
df_results.loc[:,'training time (hr)'] = df_results['training time (s)']/3600
# remove regressor from names
df_results['algorithm'] = df_results['algorithm'].apply(lambda x: x.replace('Regressor','')) 
#Rename SGD to Linear
df_results['algorithm'] = df_results['algorithm'].apply(lambda x: 'Linear' if x=='SGD' else x)
# rename sembackpropgp to SBP
df_results['algorithm'] = df_results['algorithm'].apply(lambda x: x.replace('sembackpropgp','SBP-GP'))
# rename FE_AFP to AFP_FE
df_results['algorithm'] = df_results['algorithm'].apply(lambda x: x.replace('FE_AFP','AFP_FE'))
# rename GPGOMEA to GP-GOMEA
df_results['algorithm'] = df_results['algorithm'].apply(lambda x: x.replace('GPGOMEA','GP-GOMEA'))
# add modified R2 with 0 floor
df_results['r2_zero_test'] = df_results['r2_test'].apply(lambda x: max(x,0))
# label friedman ddatasets
df_results.loc[:,'friedman_dataset'] = df_results['dataset'].str.contains('_fri_')
print('loaded',len(df_results),'results')
# additional metadata
df_results['symbolic_alg'] = df_results['algorithm'].apply(lambda x: x in symbolic_algs)

for col in ['algorithm','dataset']:
    print(df_results[col].nunique(), col+'s')


##########
# save results
##########
df_results.to_feather('../results/black-box_results.feather')
print('results saved to ../results/black-box_results.feather')

########
print('mean trial count:')
print(df_results.groupby('algorithm')['dataset'].count().sort_values()
      / df_results.dataset.nunique())

