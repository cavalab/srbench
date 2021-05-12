from ..gplearn import complexity,model
from gplearn.genetic import SymbolicRegressor
import re
from .params._gplearn import params

est = SymbolicRegressor(
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

est.set_params(**params)
est.function_set = ('add', 'sub', 'mul', 'div', 'log','sqrt', 'sin','cos')

# double the evals
est.population_size *= 2**0.5
est.generations *= 2**0.5
