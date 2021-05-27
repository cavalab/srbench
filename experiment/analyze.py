import pdb
import pandas as pd
import subprocess
import numpy as np
from glob import glob
import argparse
import os, errno, sys
from joblib import Parallel, delayed
from seeds import SEEDS
from yaml import load, Loader

#TODO make this script smarter about running jobs. 
# have it check to see whether results for that job exist before
# resubmitting. That way the same command can be run multiple times.

if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser(
            description="An analyst for quick ML applications.", add_help=False)
    parser.add_argument('DATASET_DIR', type=str,
                        help='Dataset directory like (pmlb/datasets)')    
    parser.add_argument('-h', '--help', action='help',
                        help='Show this help message and exit.')
    parser.add_argument('-ml', action='store', dest='LEARNERS',default=None,
            type=str, help='Comma-separated list of ML methods to use (should '
            'correspond to a py file name in methods/)')
    parser.add_argument('--local', action='store_true', dest='LOCAL', default=False, 
            help='Run locally as opposed to on LPC')
    parser.add_argument('--slurm', action='store_true', dest='SLURM', default=False, 
            help='Run on a SLURM scheduler as opposed to on LPC')
    parser.add_argument('--noskips', action='store_true', dest='NOSKIPS', default=False, 
            help='Overwite existing results if found')
    parser.add_argument('-skip_tuning', action='store_true', dest='SKIP_TUNE', default=False, 
            help='Skip tuning step')
    parser.add_argument('-A', action='store', dest='A', default='plgsrbench', 
            help='SLURM account')
    parser.add_argument('-sym_data',action='store_true', dest='SYM_DATA', default=False, 
            help='Specify a symbolic dataset')
    parser.add_argument('-tuned',action='store_true', dest='TUNED', default=False, 
            help='Run tuned version of estimators. Only applies when ml=None')
    parser.add_argument('-n_jobs',action='store',dest='N_JOBS',default=1,type=int,
            help='Number of parallel jobs')
    parser.add_argument('-time_limit',action='store',dest='TIME',default='48:00',
            type=str, help='Time limit (hr:min) e.g. 24:00')
    parser.add_argument('-seed',action='store',dest='SEED',default=None,
            type=int, help='A specific random seed')
    parser.add_argument('-n_trials',action='store',dest='N_TRIALS',default=1,
            type=int, help='Number of parallel jobs')
    parser.add_argument('-label',action='store',dest='LABEL',default='class',
            type=str,help='Name of class label column')
    parser.add_argument('-results',action='store',dest='RDIR',default='results',
            type=str,help='Results directory')
    parser.add_argument('-q',action='store',dest='QUEUE',
                        default='epistasis_long',
                        type=str,help='LSF queue')
    parser.add_argument('-script',action='store',dest='SCRIPT',
                        default='evaluate_model',
                        type=str,help='Python script to run')
    parser.add_argument('-m',action='store',dest='M',default=16384,type=int,
            help='LSF memory request and limit (MB)')
    parser.add_argument('-starting_seed',action='store',dest='START_SEED',
                        default=0,type=int, help='seed position to start with')
    parser.add_argument('-test',action='store_true', dest='TEST', 
                       help='Used for testing a minimal version')
    parser.add_argument('-target_noise',action='store',dest='Y_NOISE',
                        default=0.0, type=float, help='Gaussian noise to add'
                        'to the target')
    parser.add_argument('-feature_noise',action='store',dest='X_NOISE',
                        default=0.0, type=float, help='Gaussian noise to add'
                        'to the target')
    parser.add_argument('-job_limit',action='store',dest='JOB_LIMIT',
                        default=1000, type=int, 
                        help='Limit number of jobs submitted at once')

    args = parser.parse_args()
     
    if args.SLURM and args.QUEUE == 'epistasis_long':
        print('setting queue to plgrid')
        args.QUEUE = 'plgrid'

    if args.LEARNERS == None:
        prefix = 'methods/tuned/' if args.TUNED else 'methods/'
        learners = [ml.split('/')[-1][:-3] for ml in glob(prefix+'*.py') 
                if not ml.split('/')[-1].startswith('_')]
        if args.TUNED:
            learners = ['tuned.'+ml for ml in learners]
    else:
        learners = [ml for ml in args.LEARNERS.split(',')]  # learners
    print('learners:',learners)

    print('dataset directory:',args.DATASET_DIR)

    if args.DATASET_DIR.endswith('.tsv.gz'):
        datasets = [args.DATASET_DIR]
    elif args.DATASET_DIR.endswith('*'):
        print('capturing glob',args.DATASET_DIR+'/*.tsv.gz')
        datasets = glob(args.DATASET_DIR+'*/*.tsv.gz')
    else:
        datasets = glob(args.DATASET_DIR+'/*/*.tsv.gz')
    print('found',len(datasets),'datasets')

    #####################################################
    ## look for existing jobs
    ########################
    current_jobs = []
    if args.SLURM:
        res = subprocess.check_output(['squeue -o "%j"'],shell=True)
        current_jobs = res.decode().split('\n')
    elif not args.LOCAL:
        res = subprocess.check_output(['bjobs -o "JOB_NAME" -noheader'],shell=True)
        current_jobs = res.decode().split('\n')
    # current_jobs = ['_'.join(cj.split('_')[:-1]) for cj in current_jobs]

    # write run commands
    jobs_w_results = []
    jobs_wout_results = [] 
    suffix = ('.json.updated' if args.SCRIPT=='assess_symbolic_model' else
                  '.json')
    queued_jobs = []
    all_commands = []
    job_info=[]
    for t in range(args.START_SEED, args.START_SEED+args.N_TRIALS):
        # random_state = np.random.randint(2**15-1)
        if args.SEED and args.N_TRIALS==1:
            random_state = args.SEED
        else:
            random_state = SEEDS[t]
        # print('random_seed:',random_state)
        for dataset in datasets:
            if (not args.SYM_DATA 
                and any([n in dataset for n in ['feynman', 'strogatz']])
               ):
                continue
            # grab regression datasets
            metadata = load(
                open('/'.join(dataset.split('/')[:-1])+'/metadata.yaml','r'),
                    Loader=Loader)
            if metadata['task'] != 'regression':
                continue
            
            dataname = dataset.split('/')[-1].split('.tsv.gz')[0]
            results_path = '/'.join([args.RDIR, dataname]) + '/'
            if not os.path.exists(results_path):
                os.makedirs(results_path)
                
            for ml in learners:
                save_file = (results_path + '/' + dataname + '_' + ml + '_' 
                             + str(random_state))
                # if updated, check if json file exists (required)
                if ('updated' in suffix 
                    or args.SCRIPT.startswith('fix_')):
                    if not os.path.exists(save_file+'.json'):
                        jobs_wout_results.append([save_file,'json result DNE'])
                        continue

                if not args.NOSKIPS:
                    save_file = (results_path + '/' + dataname + '_' + ml + '_' 
                                 + str(random_state))
                    if args.Y_NOISE > 0:
                        save_file += '_target-noise'+str(args.Y_NOISE)
                    if args.X_NOISE > 0:
                        save_file += '_feature-noise'+str(args.X_NOISE)

                    # check if there is already a result for this experiment
                    if (os.path.exists(save_file+suffix) 
                        and args.SCRIPT != 'fix_aifeynman_model_size'):
                        jobs_w_results.append([save_file,'exists'])
                        continue
                    # check if there is already a queued job for this experiment
                    if save_file.split('/')[-1] in current_jobs:
                        queued_jobs.append([save_file,'queued'])
                        continue

                
                all_commands.append('python {SCRIPT}.py '
                                    '{DATASET}'
                                    ' -ml {ML}'
                                    ' -results_path {RDIR}'
                                    ' -seed {RS} '
                                    ' -target_noise {TN} '
                                    ' -feature_noise {FN} '
                                    '{TEST} {SYM_DATA} {SKIP_TUNE}'.format(
                                        SCRIPT=args.SCRIPT,
                                        ML=ml,
                                        DATASET=dataset,
                                        RDIR=results_path,
                                        RS=random_state,
                                        TN=args.Y_NOISE,
                                        FN=args.X_NOISE,
                                        TEST=('-test' if args.TEST
                                                else ''),
                                        SYM_DATA=('-sym_data' if args.SYM_DATA
                                                   else ''),
                                        SKIP_TUNE=('-skip_tuning' if
                                                   args.SKIP_TUNE else '')
                                        )
                                    )
                job_info.append({'ml':ml,
                                 'dataset':dataname,
                                 'seed':str(random_state),
                                 'results_path':results_path,
                                 'target_noise':args.Y_NOISE
                                 })
    if len(all_commands) > args.JOB_LIMIT:
        print('shaving jobs down to job limit ({})'.format(args.JOB_LIMIT))
        all_commands = all_commands[:args.JOB_LIMIT]
    if not args.NOSKIPS:
        print('skipped',len(jobs_w_results),'jobs with results. Override with --noskips.')
        print('skipped',len(jobs_wout_results),'jobs without results. Override with --noskips.')
        print('skipped',len(queued_jobs),'queued jobs. Override with --noskips.')
    print('submitting',len(all_commands),'jobs...')
    if args.LOCAL:
        # run locally  
        for run_cmd in all_commands: 
            print(run_cmd)
            Parallel(n_jobs=args.N_JOBS)(delayed(os.system)(run_cmd) 
                                     for run_cmd in all_commands)
            # os.system(run_cmd)
    else:
        # sbatch
        for i,run_cmd in enumerate(all_commands):
            job_name = '_'.join([
                                 job_info[i]['dataset'],
                                 job_info[i]['ml'],
                                 job_info[i]['seed'],
                                 args.SCRIPT
                                ])
            if args.Y_NOISE>0:
                job_name += '_target-noise'+str(args.Y_NOISE)
            if args.X_NOISE>0:
                job_name += '_feature-noise'+str(args.X_NOISE)
            out_file = (job_info[i]['results_path']
                        + job_name 
                        + '.%J.out')
            error_file = out_file[:-4] + '.err'
            
            if args.SLURM:
                    batch_script = \
"""#!/usr/bin/bash 
#SBATCH -o {OUT_FILE} 
#SBATCH --error={ERR_FILE} 
#SBATCH -N 1 
#SBATCH -n {N_CORES} 
#SBATCH -J {JOB_NAME} 
#SBATCH -A {A} -p {QUEUE} 
#SBATCH --ntasks-per-node=1 --time={TIME}:00 
#SBATCH --mem-per-cpu={M} 

conda info 
source plg_modules.sh

{cmd}
""".format(
           OUT_FILE=out_file,
           ERR_FILE=error_file,
           JOB_NAME=job_name,
           QUEUE=args.QUEUE,
           A=args.A,
           N_CORES=args.N_JOBS,
           M=args.M,
           cmd=run_cmd,
           TIME=args.TIME
          )
                    with open('tmp_script','w') as f:
                        f.write(batch_script)

                    # print(batch_script)
                    print(job_name)
                    sbatch_response = subprocess.check_output(['sbatch tmp_script'],
                                                              shell=True).decode()     # submit jobs 
                    print(sbatch_response)

            else: # LPC
                # activate srbench env, load modules
                # pre_run_cmds = ["conda activate srbench",
                #                 "source lpc_modules.sh"]
                # run_cmd = '; '.join(pre_run_cmds + [run_cmd])
                bsub_cmd = ('bsub -o {OUT_FILE} '
                            '-e {ERR_FILE} '
                            '-n {N_CORES} '
                            '-J {JOB_NAME} '
                            '-q {QUEUE} '
                            '-R "span[hosts=1] rusage[mem={M}]" '
                            '-W {TIME} '
                            '-M {M} ').format(
                                   OUT_FILE=out_file,
                                   ERR_FILE=error_file,
                                   JOB_NAME=job_name,
                                   QUEUE=args.QUEUE,
                                   N_CORES=args.N_JOBS,
                                   M=args.M,
                                   TIME=args.TIME
                                   )
                
                bsub_cmd +=  '"' + run_cmd + '"'
                print(bsub_cmd)
                os.system(bsub_cmd)     # submit jobs 

    print('Finished submitting',len(all_commands),'jobs.')
