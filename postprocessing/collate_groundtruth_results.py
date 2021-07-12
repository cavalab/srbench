"""Collates json-formatted results, cleans them up and saves them as .feather
files."""
# Author: William La Cava, williamlacava@gmail.com
# SRBENCH
# License: GPLv3

################################################################################
# Ground-truth problems
################################################################################
import pandas as pd
import json
import numpy as np
from glob import glob
from tqdm import tqdm
import os
import sys

rdir = '../results_sym_data/'
if len(sys.argv) > 1:
    rdir = sys.argv[1]
else:
    print('no rdir provided, using',rdir)
print('reading results from  directory', rdir)

##########
# load data from json
##########

frames = []
excluded_datasets = [
    'feynman_test_10',
    'feynman_I_26_2',
    'feynman_I_30_5'
]
excluded_cols = [
    'params'
]
fails = []
bad_bsr = []
updated = 0
for f in tqdm(glob(rdir + '/*/*.json')):
    if os.path.exists(f+'.updated'):
        f += '.updated'
        updated += 1
    if 'cv_results' in f: 
        continue
    if 'EHC' in f:
        continue
    if any([ed in f for ed in excluded_datasets]):
        continue
    try: 
        r = json.load(open(f,'r'))
        if isinstance(r['symbolic_model'],list):
            print('WARNING: list returned for model:',f)
            bad_bsr.append(f)
            sm = ['B'+str(i)+'*'+ri for i, ri in enumerate(r['symbolic_model'])]
            sm = '+'.join(sm)
            r['symbolic_model'] = sm
            
        sub_r = {k:v for k,v in r.items() if k not in excluded_cols}
    #     df = pd.DataFrame(sub_r)
        frames.append(sub_r) 
    #     print(f)
    #     print(r.keys())
    except Exception as e:
        fails.append([f,e])
        pass
    
print('{} results files loaded, {} ({:.1f}%) of which are '
	'updated'.format(len(frames), updated, updated/len(frames)*100))
print(len(fails),'fails:')
for f in fails: 
    print(f[0])
print('bad bsr:',bad_bsr)
df_results = pd.DataFrame.from_records(frames)
##########
# cleanup
##########
df_results = df_results.rename(columns={'time_time':'training time (s)'})
df_results.loc[:,'training time (hr)'] = df_results['training time (s)']/3600
# add modified R2 with 0 floor
df_results['r2_zero_test'] = df_results['r2_test'].apply(lambda x: max(x,0))
for col in ['symbolic_error_is_zero', 'symbolic_error_is_constant', 'symbolic_fraction_is_constant']:
    df_results.loc[:,col] = df_results[col].fillna(False)
print(','.join(df_results.algorithm.unique()))
# remove 'Regressor' from names
df_results['algorithm'] = df_results['algorithm'].apply(lambda x: x.replace('Regressor','')) 
df_results['algorithm'] = df_results['algorithm'].apply(lambda x: x.replace('tuned.','')) 
df_results['algorithm'] = df_results['algorithm'].apply(lambda x: x.replace('sembackpropgp','SBP-GP')) 
# rename FE_AFP to AFP_FE
df_results['algorithm'] = df_results['algorithm'].apply(lambda x: x.replace('FE_AFP','AFP_FE'))
# rename GPGOMEA to GP-GOMEA
df_results['algorithm'] = df_results['algorithm'].apply(lambda x: x.replace('GPGOMEA','GP-GOMEA'))
# indicator of strogatz or feynman
df_results['data_group'] = df_results['dataset'].apply(lambda x: 'Feynman' if 'feynman' in x else 'Strogatz') 

##########
# compute symbolic solutions
##########
df_results.loc[:,'symbolic_solution'] = df_results[['symbolic_error_is_zero',
                                                    'symbolic_error_is_constant',
                                                    'symbolic_fraction_is_constant']
                                                   ].apply(any,raw=True, axis=1)
df_results.loc[:,'symbolic_solution'] = df_results['symbolic_solution'] & ~df_results['simplified_symbolic_model'].isna() 
df_results.loc[:,'symbolic_solution'] = df_results['symbolic_solution'] & ~(df_results['simplified_symbolic_model'] == '0')
df_results.loc[:,'symbolic_solution'] = df_results['symbolic_solution'] & ~(df_results['simplified_symbolic_model'] == 'nan')

##########
# save results
##########
df_results.to_feather('../results/ground-truth_results.feather')
print('results saved to ../results/ground-truth_results.feather')

