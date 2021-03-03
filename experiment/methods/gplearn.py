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

est = gplearn.SymbolicRegressor(function_set=('add', 'sub', 'mul', 'div','max','min','log','sqrt'),
                                tournament_size=20,
                                init_depth=(2, 6),
                                init_method='half and half',
                                metric='mean absolute error',
                                parsimony_coefficient=0.001,
                                p_crossover=0.9,
                                p_subtree_mutation=0.01, p_hoist_mutation=0.01, p_point_mutation=0.01, p_point_replace=0.05,
                                max_samples=1.0)

)

def model(est):
    return str(est._program)

def complexity(est):
    #TODO: check
    return len(re.split('\(|,',model(est)))