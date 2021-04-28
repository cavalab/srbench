from .src.mrgp import MRGPRegressor

hyper_params = {}
# hyper_params = [
#     {
#         'popsize': (100,), 'g': (1000,),
#         'max_len': (6,),
#         'rt_cross':(0.2,),'rt_mut':(0.8,),
#     },
#     {
#         'popsize': (100,), 'g': (1000,),
#         'max_len': (6,),
#         'rt_cross':(0.8,),'rt_mut':(0.2,),
#     },
#     {
#         'popsize': (100,), 'g': (1000,),
#         'max_len': (6,),
#         'rt_cross':(0.5,),'rt_mut':(0.5,),
#     },
#     {
#         'popsize': (1000,), 'g': (100,),
#         'max_len': (6,),
#         'rt_cross':(0.2,),'rt_mut':(0.8,),
#     },
#     {
#         'popsize': (1000,), 'g': (100,),
#         'max_len': (6,),
#         'rt_cross':(0.8,),'rt_mut':(0.2,),
#     },
#     {
#         'popsize': (1000,), 'g': (100,),
#         'max_len': (6,),
#         'rt_cross':(0.5,),'rt_mut':(0.5,),
#     },
# ]

est=MRGPRegressor()

def complexity(est):
    return est.complexity

def model(est):
    return str(est.model_)

