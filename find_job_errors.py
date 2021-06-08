import pdb
from glob import glob
import os
import subprocess
import time

# for f in glob('results_sym_data/*/*FE_AFP*.json'):
#     out=glob(f[:-5]+'*.out')
#     print(len(out),'out files')
# aggregate job names
job_names = []
for f in glob('results_sym_data/*/*.out'):
    if 'assess' in f: continue
    job_name = '_'.join(f.split('_')[:-1])
    if job_name not in job_names:
        res=glob(job_name+'*.json')
        if len(res) == 0:
            job_names.append(job_name)
print(80*'=')
print('Missing results for:')
for j in job_names:
    print(j)
print(80*'=')
# print('Errors:')
# # check for results
# for j in job_names: #[:10]:
#     res=glob(j+'*.json')
#     if len(res) == 0:
#         # print(job_name)
#         # find newest outfile
#         out_files = glob(j+'*.out')
#         latest = 0
#         latest_idx = 0
#         ts = []
#         for i,f in enumerate(out_files):
#             t = os.path.getmtime(f)
#             # print(time.ctime(t))
#             ts.append(ts)
#             if t > latest:
#                 latest = t
#                 latest_idx = i

#         latest_run = out_files[latest_idx]
#         print(60*'-')
#         print('{:s}\n{:s}'.format(time.ctime(os.path.getmtime(latest_run)),
#                                   latest_run)
#              )
#         error = subprocess.run('grep "error" -i {}'.format( latest_run), 
#                                 shell=True,
#                                stdout=subprocess.PIPE).stdout
#         if len(error)==0:
#             error = subprocess.run('tail {}'.format( latest_run), shell=True,
#                                    stdout=subprocess.PIPE).stdout
#         print('\t',error.decode())
#         print(60*'-')

