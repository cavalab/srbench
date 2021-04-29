from gplearn.genetic import SymbolicRegressor
import re

hyper_params = []
for p, g in zip([1000,500,100],[500,1000,5000]):
    for fs in [('add', 'sub', 'mul', 'div','exp','log','sqrt'),
               ('add', 'sub', 'mul', 'div','sin','cos','exp','log','sqrt')]:
        hyper_params.append({
            'population_size' : [p],
            'generations' : [g],
            'function_set': [fs]
            })

est = SymbolicRegressor(function_set=('add', 'sub', 'mul', 'div','max','min','log','sqrt'),
                        tournament_size=20,
                        init_depth=(2, 6),
                        init_method='half and half',
                        metric='mean absolute error',
                        parsimony_coefficient=0.001,
                        p_crossover=0.9,
                        p_subtree_mutation=0.01, 
                        p_hoist_mutation=0.01, 
                        p_point_mutation=0.01, 
                        p_point_replace=0.05,
                        max_samples=1.0
                       )


def model(est):
    return str(est._program)

def complexity(est):
    #TODO: check
    return len(re.split('\(|,',model(est)))
