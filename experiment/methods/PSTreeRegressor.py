from deap.tools import selRandom
from pstree.cluster_gp_sklearn import GPRegressor, PSTreeRegressor, selTournamentDCD
from pstree.complexity_utils import tree_gp_regressor_complexity
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import ParameterSampler

hyper_params = list(ParameterSampler({
    'basic_primitive': ['optimal', 'log,sqrt,sin,tanh'],
    'initial_height': [None, '0-6'],
    'max_leaf_nodes': [4, 6, 8],
}, n_iter=6, random_state=0))

for g in hyper_params:
    g['select'] = selRandom if g['initial_height'] is None else selTournamentDCD

est = PSTreeRegressor(regr_class=GPRegressor, tree_class=DecisionTreeRegressor,
                      height_limit=6, n_pop=25, n_gen=500, normalize=False,
                      basic_primitive='optimal', size_objective=True)


def complexity(est):
    _, _, total_complexity, _ = tree_gp_regressor_complexity(est)
    return total_complexity


def model(e):
    return e.model()
