import sys
import metrics
from .gsgp import GSGPClassifier

# read arguments (1: dataset name, 2: output file name, 3: trial #, 4: # cores)
dataset = sys.argv[1]
savefile = sys.argv[2]
random_state = sys.argv[3]


hyper_params = [
    {
        'popsize': (100,), 'g': (1000,),
        'max_len': (6,),
        'rt_cross':(0.0,),'rt_mut':(1.0,),
    },
    {
        'popsize': (100,), 'g': (1000,),
        'max_len': (6,),
        'rt_cross':(0.1,),'rt_mut':(0.9,),
    },
    {
        'popsize': (100,), 'g': (1000,),
        'max_len': (6,),
        'rt_cross':(0.2,),'rt_mut':(0.8,),
    },
    {
        'popsize': (1000,), 'g': (100,),
        'max_len': (6,),
        'rt_cross':(0.0,),'rt_mut':(1.0,),
    },
    {
        'popsize': (1000,), 'g': (100,),
        'max_len': (6,),
        'rt_cross':(0.1,),'rt_mut':(0.9,),
    },
    {
        'popsize': (1000,), 'g': (100,),
        'max_len': (6,),
        'rt_cross':(0.2,),'rt_mut':(0.8,),
    },
]


est=GSGPClassifier(dataset=dataset.split('/')[-1][:-7], y_test=y_test, y_train=y_test)

from .evaluate_model import evaluate_model
evaluate_model(dataset, save_file, random_state, est, hyper_params)


