if (($#==1)); #check if number of arguments is 1 
then
    mls="-ml ${1}"
else
    mls=''
fi

python analyze.py data/stage0/ -results ../results_stage0 -n_jobs 4 -m 16384 -n_trials 5 $mls
