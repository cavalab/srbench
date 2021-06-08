import pandas as pd
import numpy as np
from glob import glob
import argparse
import os, errno, sys
from joblib import Parallel, delayed
from seeds import SEEDS
from yaml import load, Loader


if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser(
            description="Empirical running time analysis", add_help=False)
    parser.add_argument('DATASET', type=str,
                        help='Dataset')    
    parser.add_argument('-h', '--help', action='help',
                        help='Show this help message and exit.')
    parser.add_argument('-ml', action='store', dest='LEARNERS',default=None,
            type=str, help='Comma-separated list of ML methods to use (should '
            'correspond to a py file name in methods/). None means all')
    parser.add_argument('-seed',action='store',dest='SEED',default=None,
            type=int, help='A specific random seed')
    parser.add_argument('-n_trials',action='store',dest='N_TRIALS',default=1,
            type=int, help='Number of parallel jobs')
    parser.add_argument('-results',action='store',dest='RDIR',default='results',
            type=str,help='Results directory')
    parser.add_argument('-test',action='store_true', dest='TEST', 
                       help='Used for testing a minimal version')
    parser.add_argument('-target_noise',action='store',dest='Y_NOISE',
                        default=0.0, type=float, help='Gaussian noise to add'
                        'to the target')
    parser.add_argument('-feature_noise',action='store',dest='X_NOISE',
                        default=0.0, type=float, help='Gaussian noise to add'
                        'to the target')

    args = parser.parse_args()
     
    if args.LEARNERS == None:
        learners = [ml.split('/')[-1][:-3] for ml in glob('methods/*.py') 
                if not ml.split('/')[-1].startswith('_')]
    else:
        learners = [ml for ml in args.LEARNERS.split(',')]  # learners
    print('learners:',learners)

    print('dataset:',args.DATASET)

    dataset = args.DATASET
    # write run commands
    all_commands = []
    job_info=[]
    for t in range(args.N_TRIALS):
        # random_state = np.random.randint(2**15-1)
        if args.SEED != None:
            assert args.N_TRIALS == 1
            random_state = args.SEED
        else:
            random_state = SEEDS[t]
            
        dataname = dataset.split('/')[-1].split('.tsv.gz')[0]
        results_path = '/'.join([args.RDIR, dataname]) + '/'
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        
        for ml in learners:
            
            all_commands.append('python evaluate_model.py '
                                '{DATASET}'
                                ' -ml {ML}'
                                ' -results_path {RDIR}'
                                ' -seed {RS} '
                                ' -target_noise {TN} '
                                ' -feature_noise {FN} '
                                '{TEST} {SKIP}'.format(
                                    ML=ml,
                                    DATASET=dataset,
                                    RDIR=results_path,
                                    RS=random_state,
                                    TN=args.Y_NOISE,
                                    FN=args.X_NOISE,
                                    TEST=('-test' if args.TEST
                                            else '')
                                    SKIP='-skip_tuning'
                                    )
                                )
            job_info.append({'ml':ml,
                             'dataset':dataname,
                             'seed':str(random_state),
                             'results_path':results_path})

    # if args.LOCAL:
    # run locally  
    for run_cmd in all_commands: 
        print(run_cmd)
        os.system(run_cmd)
        # Parallel(n_jobs=args.N_JOBS)(delayed(os.system)(run_cmd) 
        #                          for run_cmd in all_commands)
    # else: # LPC
    #     for i,run_cmd in enumerate(all_commands):
    #         job_name = '_'.join([
    #                              job_info[i]['dataset'],
    #                              job_info[i]['ml'],
    #                              job_info[i]['seed']
    #                             ])
    #         out_file = job_info[i]['results_path'] + job_name + '_%J.out'
    #         error_file = out_file[:-4] + '.err'
            
    #         bsub_cmd = ('bsub -o {OUT_FILE} -n {N_CORES} -J {JOB_NAME} -q {QUEUE} '
    #                    '-R "span[hosts=1] rusage[mem={M}]" -M {M} ').format(
    #                            OUT_FILE=out_file,
    #                            JOB_NAME=job_name,
    #                            QUEUE=args.QUEUE,
    #                            N_CORES=args.N_JOBS,
    #                            M=args.M)
            
    #         bsub_cmd +=  '"' + run_cmd + '"'
    #         print(bsub_cmd)
    #         os.system(bsub_cmd)     # submit jobs 
