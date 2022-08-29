import pdb
import pandas as pd
import subprocess
import numpy as np
from glob import glob
import argparse
import os, errno, sys
# from joblib import Parallel, delayed
from seeds import SEEDS
from yaml import load, Loader
from utils import Competitors, FilteredCompetitors

if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser(
            description="An analyst for quick ML applications.", add_help=False)
    parser.add_argument('DATASET_DIR', type=str, action='extend', nargs='+',
                        help='Dataset directory with datasets, or datasets '
                        'themselves')    
    parser.add_argument('-h', '--help', action='help',
                        help='Show this help message and exit.')
    parser.add_argument('-ml', action='store', dest='LEARNERS',default=None,
            type=str, help='Comma-separated list of ML methods to use (should '
            'correspond to a py file name in methods/)')
    parser.add_argument('--local', action='store_true', dest='LOCAL', default=False, 
            help='Run locally as opposed to on LPC')
    parser.add_argument('--noskips', action='store_true', dest='NOSKIPS', default=False, 
            help='Overwite existing results if found')
    parser.add_argument('-skip_tuning', action='store_true', dest='SKIP_TUNE', default=False, 
            help='Skip tuning step')
    parser.add_argument('-max_samples',action='store',  type=int, default=0,
                        help='number of training samples')
    parser.add_argument('-tuned',action='store_true', dest='TUNED', default=False, 
            help='Run tuned version of estimators. Only applies when ml=None')
    parser.add_argument('-n_jobs',action='store',dest='N_JOBS',default=4,type=int,
            help='Number of parallel jobs')
    parser.add_argument('-time_limit',action='store',dest='TIME',default=None,
            type=str, help='Time limit (hr:min) e.g. 24:00')
    parser.add_argument('-seed',action='store',dest='SEED',default=None,
            type=int, help='A specific random seed')
    parser.add_argument('-n_trials',action='store',dest='N_TRIALS',default=1,
            type=int, help='Number of repeat experiments with different seed')
    parser.add_argument('-label',action='store',dest='LABEL',default='class',
            type=str,help='Name of class label column')
    parser.add_argument('-results',action='store',dest='RDIR',default='../results',
            type=str,help='Results directory')
    parser.add_argument('-q',action='store',dest='QUEUE',
                        default=None,
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
    parser.add_argument('-job_limit',action='store',dest='JOB_LIMIT',
                        default=5000, type=int, 
                        help='Limit number of jobs submitted at once')
    parser.add_argument('-dry_run',action='store_true',default=False,
                        help='Just print')
    parser.add_argument('-file_ending',action='store',default='tsv.gz',
                        help='dataset file ending')
    parser.add_argument('-stage',action='store', type=int, default=0,  
                       help='Competition stage')

    args = parser.parse_args()
     
    if args.LEARNERS == None:
        learners = Competitors if args.stage==0 else FilteredCompetitors
    else:
        learners = [ml for ml in args.LEARNERS.split(',')]  # learners
    print('learners:',learners)

    print('dataset directory:',args.DATASET_DIR)

    if len(args.DATASET_DIR) > 1:
        datasets = args.DATASET_DIR
    else:
        datadir = args.DATASET_DIR[0]
        if datadir.endswith(args.file_ending):
            if ',' in datadir:
                datasets = args.DATASET_DIR.split(',')
                print(datasets)
            else:
                datasets = [datadir]
        else:
            print('capturing glob',datadir+f'/*.{args.file_ending}')
            datasets = glob(datadir+f'/*.{args.file_ending}')
    print('found',len(datasets),'datasets')

    file_ending = '.'.join(datasets[0].split('.')[1:])
    #####################################################
    ## look for existing jobs
    ########################
    current_jobs = []
    if not args.LOCAL and not args.dry_run:
        res = subprocess.check_output(['bjobs -o "JOB_NAME" -noheader'],
                                      shell=True)
        current_jobs = res.decode().split('\n')
    # current_jobs = [f'srbench_{cj}_{args.SCRIPT}' for cj in current_jobs]

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
            ########################################
            # set max time
            df = pd.read_csv(dataset)
            if args.TIME == None:
                if len(df) <= 1000: 
                    time_limit = '1:05'
                    if args.QUEUE==None: 
                        queue = 'epistasis_normal' 
                    else:
                        queue = args.QUEUE
                else:
                    time_limit = '10:05'
                    if args.QUEUE==None: 
                        queue = 'epistasis_long' 
                    else:
                        queue = args.QUEUE
            else:
                time_limit = args.TIME
                queue = 'epistasis_long'
            ########################################
            
            dataname = dataset.split('/')[-1].split(f'.{file_ending}')[0]
                
            for ml in learners:
                results_path = os.path.join(args.RDIR, dataname, ml) 
                if not os.path.exists(results_path):
                    os.makedirs(results_path, exist_ok=True)

                save_file = os.path.join(results_path,
                        '_'.join([dataname, ml, str(random_state)])
                        )


                # if updated, check if json file exists (required)
                if ('updated' in suffix 
                    or args.SCRIPT.startswith('fix_')):
                    if not os.path.exists(save_file+'.json'):
                        jobs_wout_results.append([save_file,'json result DNE'])
                        continue

                if not args.NOSKIPS:
                    # check if there is already a result for this experiment
                    if (os.path.exists(save_file+suffix) ):
                        jobs_w_results.append([save_file,'exists'])
                        continue
                    # check if there is already a queued job for this experiment
                    tmp = save_file.split('/')[-1]
                    jobname = f'srbench_{tmp}_{args.SCRIPT}' 
                    if jobname in current_jobs:
                        queued_jobs.append([save_file,'queued'])
                        continue

                
                all_commands.append(
                        # 'eval "$(conda shell.bash hook)"\n '
                        # 'conda init bash\n'
                        # 'conda activate srcomp-{ML}\n '
                        # 'conda info\n '
                        f'conda run -n srcomp-{ml} '
                        f'python {args.SCRIPT}.py'
                        f' {dataset}'
                        f' -ml {ml}'
                        f' -results_path {results_path}'
                        f' -seed {random_state} '
                        f' -stage {args.stage}'
                        f' -max_samples {args.max_samples}'
                        f" {'-test' if args.TEST else ''}"
                        # .format(
                        #     SCRIPT=args.SCRIPT,
                        #     ML=ml,
                        #     DATASET=dataset,
                        #     RDIR=results_path,
                        #     RS=random_state,
                        #     TN=args.Y_NOISE,
                        #     FN=args.X_NOISE,
                        #     TEST=('-test' if args.TEST else ''),
                        #     SYM_DATA=('-sym_data' if args.SYM_DATA else ''))
                )

                job_info.append({'ml':ml,
                                 'dataset':dataname,
                                 'seed':str(random_state),
                                 'results_path':results_path,
                                 'time_limit':time_limit,
                                 'queue':queue
                                 })
    if len(all_commands) > args.JOB_LIMIT:
        print('shaving jobs down to job limit ({})'.format(args.JOB_LIMIT))
        all_commands = all_commands[:args.JOB_LIMIT]
    if not args.NOSKIPS:
        if len(jobs_w_results)>0:
            print('skipped',len(jobs_w_results),'jobs with results. Override with --noskips.')
        if len(jobs_wout_results)>0:
            print('skipped',len(jobs_wout_results),'jobs without results. Override with --noskips.')
        if len(queued_jobs)>0:
            print('skipped',len(queued_jobs),'queued jobs. Override with --noskips.')
    ############################################################################
    # job submission
    print('submitting',len(all_commands),'jobs...')
    if args.LOCAL:
        # run locally  
        if args.dry_run:
            for run_cmd in all_commands:
                print(run_cmd)
        # else:
        #     Parallel(n_jobs=args.N_JOBS)(delayed(os.system)(run_cmd)
        #                          for run_cmd in all_commands)
    else:
        # LSF 
        for i,run_cmd in enumerate(all_commands):
            job_name = '_'.join(['srbench',
                                 job_info[i]['dataset'],
                                 job_info[i]['ml'],
                                 job_info[i]['seed'],
                                 args.SCRIPT
                                ])
            time_limit = job_info[i]['time_limit']
            queue = job_info[i]['queue']

            out_file = os.path.join(job_info[i]['results_path'],
                                    job_name + '.out')
            # if out_file exists, remove it
            if os.path.exists(out_file):
                os.remove(out_file)
            # error_file = out_file[:-4] + '.err'

            ML = job_info[i]['ml']
            bsub_cmd = (f'bsub -o {out_file} '
                        # f'-e {error_file} '
                        f'-J {job_name} '
                        f'-n {args.N_JOBS} '
                        f'-q {queue} '
                        f'-R "span[hosts=1] rusage[mem={args.M}]" '
                        f'-W {time_limit} '
                        f'-M {args.M} '
                        )

            bsub_cmd +=  '"' + run_cmd + '"'
            if args.dry_run:
                print(bsub_cmd)
            else:
                print(bsub_cmd)
                os.system(bsub_cmd)     # submit jobs 

    print('Finished submitting',len(all_commands),'jobs.')
