from glob import glob
import os
import sys
import argparse
from pmlb import regression_dataset_names

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Submit long jobs.",
                                     add_help=False)
    parser.add_argument('DATA_PATH',type=str)
    parser.add_argument('-ml',action='store',dest='mls', type=str, 
            default='')
    parser.add_argument('-q',action='store',dest='Q', type=str, 
                        default='moore_normal')
    parser.add_argument('-n_trials',action='store',dest='TRIALS', default=1)
    parser.add_argument('-results',action='store',dest='RDIR',
            default='results',type=str,help='Results directory')
    args = parser.parse_args()

datapath = args.DATA_PATH 

if args.mls == '':
    ml_files = glob('./methods/*.py')
    mls = ','.join([m[10:-3] for m in ml_files if '__' not in m])
    print('No MLs passed. Running everything in methods/*.py:',
          mls)
else:
    mls = args.mls

lpc_options = '-q {Q} -m 12000 -n_jobs 1'.format(Q=args.Q)

for dataset in regression_dataset_names:
    f = args.DATA_PATH + '/' + dataset + '/' + dataset + '.tsv.gz' 
    jobline =  ('python analyze.py {DATA} '
               '-ml {ML} '
               '-results {RDIR} -n_trials {NT} {LPC}').format(DATA=f,
                                                      LPC=lpc_options,
                                                      ML=mls,
                                                      RDIR=args.RDIR,
                                                      NT=args.TRIALS)
    print(jobline)
    os.system(jobline)
