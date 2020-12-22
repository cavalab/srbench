import os
from glob import glob


def test_analyze_runs_with_all_methods():
    """Each method runs on a small dataset"""
    f = 'tests/192_vineyard_small.tsv.gz'

    ml_files = glob('./methods/*.py')
    mls = ','.join([m[10:-3] for m in ml_files if '__' not in m])
    print('MLs:',mls)

    jobline =  ('python analyze.py {DATA} '
               '-ml {ML} '
               '-results {RDIR} -n_trials {NT} -n_jobs 1 --local -test').format(
                   DATA=f,
                   ML=mls,
                   RDIR='tmp_results',
                   NT=1)
    print(jobline)
    os.system(jobline)

if __name__ == '__main__':
    test_analyze_runs_with_all_methods()
