from gplearn.genetic import SymbolicRegressor
import re

hyper_params = [
    {
        'population_size' : (1000,),
        'generations' : (100,),
    },
    {
        'population_size' : (500,),
        'generations' : (200,),
    },
    {
        'population_size' : (2000,),
        'generations' : (50,),
    },
]

est = gplearn.SymbolicRegressor(function_set=('add', 'sub', 'mul', 'div','max','min','log','sqrt'),)

def model(est):
    return str(est._program)

def complexity(est):
    #TODO: check
    return len(re.split('\(|,',model(est)))