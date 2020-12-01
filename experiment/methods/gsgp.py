import metrics
from .gsgp import GSGPClassifier


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


est=GSGPClassifier(dataset=dataset.split('/')[-1][:-7], y_test=y_test, 
                   y_train=y_test)

#TODO: define complexity
def complexity(est):
    return -1



