import sys
from mrgp import MRGPClassifier

# read arguments (1: dataset name, 2: output file name, 3: trial #, 4: # cores)
dataset = sys.argv[1]
savefile = sys.argv[2]
random_state = sys.argv[3]

hyper_params = [
    {
        'popsize': (100,), 'g': (1000,),
        'max_len': (6,),
        'rt_cross':(0.2,),'rt_mut':(0.8,),
    },
    {
        'popsize': (100,), 'g': (1000,),
        'max_len': (6,),
        'rt_cross':(0.8,),'rt_mut':(0.2,),
    },
    {
        'popsize': (100,), 'g': (1000,),
        'max_len': (6,),
        'rt_cross':(0.5,),'rt_mut':(0.5,),
    },
    {
        'popsize': (1000,), 'g': (100,),
        'max_len': (6,),
        'rt_cross':(0.2,),'rt_mut':(0.8,),
    },
    {
        'popsize': (1000,), 'g': (100,),
        'max_len': (6,),
        'rt_cross':(0.8,),'rt_mut':(0.2,),
    },
    {
        'popsize': (1000,), 'g': (100,),
        'max_len': (6,),
        'rt_cross':(0.5,),'rt_mut':(0.5,),
    },
]

est=MRGPClassifier(dataset=dataset.split('/')[-1][:-7])

from .evaluate_model import evaluate_model
evaluate_model(dataset, save_file, random_state, est, hyper_params)
