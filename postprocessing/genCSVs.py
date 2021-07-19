import numpy as np
import pandas as pd
from glob import glob

# CSV with datasets information

datadir = '../../../pmlb/datasets/'

frames = []
for f in glob(datadir+'/*/*.tsv.gz'):
    df = pd.read_csv(f,sep='\t') 
    group = 'feynman' if 'feynman' in f else 'strogatz' if 'strogatz' in f else 'black-box'
    frames.append(dict(
        name=f.split('/')[-1][:-7],
        nsamples = df.shape[0],
        nfeatures = df.shape[1],
        npoints = df.shape[0]*df.shape[1],
        Group=group
    ))
    
df = pd.DataFrame.from_records(frames)
df.to_csv("../docs/csv/datasets_info.csv")

# CSV for the blackbox results
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
x_vars=[ 'rmse_test',
         'log_mse_test',
         'r2_test',
         'model_size',
         'training time (s)']

df_results  = pd.read_feather('../results/black-box_results.feather')
df_blackbox = df_results.merge(df_results.groupby('dataset')['algorithm'].nunique().reset_index(),
                               on='dataset',suffixes=('','_count'))

df_sum = df_blackbox.groupby(['algorithm','dataset'],as_index=False).median()
df_sum['rmse_test'] = df_sum['mse_test'].apply(np.sqrt)
df_sum['log_mse_test'] = df_sum['mse_test'].apply(lambda x: np.log(1+x))

df_sum['algorithm'] = df_sum['algorithm'].apply(lambda x: '*'+x if x in symbolic_algs else x)
(df_sum
  .groupby(["algorithm","dataset"])
  .median()[x_vars]
  .to_csv("../docs/csv/blackbox_results.csv")
)

# Aggregate
df_sum.merge(df, left_on="dataset", right_on="name").to_csv("../docs/csv/blackbox_results_datasets.csv")

# Ground Truth results
df_results = pd.read_feather('../results/ground-truth_results.feather')
df_results.loc[:,'symbolic_solution'] = df_results[['symbolic_error_is_zero',
                                                    'symbolic_error_is_constant',
                                                    'symbolic_fraction_is_constant']
                                                   ].apply(any,raw=True, axis=1)
# clean up any corner cases (constant models, failures)
df_results.loc[:,'symbolic_solution'] = df_results['symbolic_solution'] & ~df_results['simplified_symbolic_model'].isna()
df_results.loc[:,'symbolic_solution'] = df_results['symbolic_solution'] & ~(df_results['simplified_symbolic_model'] == '0')
df_results.loc[:,'symbolic_solution'] = df_results['symbolic_solution'] & ~(df_results['simplified_symbolic_model'] == 'nan')
df_results2 = df_results.merge(df_results.groupby(['dataset','target_noise'])['algorithm'].nunique().reset_index(),
                              on=['dataset','target_noise'],suffixes=('','_count'))
# count repeat trials
df_results2 = df_results2.merge(
           df_results2.groupby(['algorithm','dataset','target_noise'])['random_state'].nunique().reset_index(),
           on=['algorithm','dataset','target_noise'],suffixes=('','_repeats'))

# accuracy-based exact solutions 
df_results2['accuracy_solution'] = df_results2['r2_test'].apply(lambda x: x > 0.999).astype(float)

# get mean solution rates for algs on datasets at specific noise levels, averaged over trials 
for soln in ['accuracy_solution','symbolic_solution']:
    df_results2 = df_results2.merge(
        df_results2.groupby(['algorithm','dataset','target_noise'])[soln].mean().reset_index(),
                                  on=['algorithm','dataset', 'target_noise'],suffixes=('','_rate'))
                                       
df_sum = df_results2.groupby(['algorithm','dataset','target_noise','data_group'],as_index=False).median()

for soln in ['accuracy_solution','symbolic_solution']:
    df_sum[soln +'_rate_(%)'] = df_sum[soln+'_rate'].apply(lambda x: x*100)
df_sum['rmse_test'] = df_sum['mse_test'].apply(np.sqrt)
df_sum['log_mse_test'] = df_sum['mse_test'].apply(lambda x: np.log(1+x))

columns = ['algorithm','dataset','target_noise','data_group'
           ,'symbolic_solution_rate_(%)','accuracy_solution'
           ,'rmse_test', 'log_mse_test'
          ]
df_sum[columns].to_csv('../docs/csv/groundtruth.csv', index=False)

