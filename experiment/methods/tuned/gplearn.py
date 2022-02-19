from ..gplearnRegressor import complexity,model, est
from .params._gplearn import params

est.set_params(**params)
est.function_set = ('add', 'sub', 'mul', 'div', 'log','sqrt', 'sin','cos')

# double the evals
est.population_size = int(est.population_size*2**0.5)
est.generations = int(est.generations*2**0.5)
