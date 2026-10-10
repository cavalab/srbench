from __future__ import annotations

from typing import Any

from pstree.cluster_gp_sklearn import GPRegressor, PSTreeRegressor, selTournamentDCD
from pstree.complexity_utils import tree_gp_regressor_complexity
from sklearn.tree import DecisionTreeRegressor

# hyper_params = list(ParameterSampler({
#     'basic_primitive': ['optimal', 'log,sqrt,sin,tanh'],
#     'initial_height': [None, '0-6'],
#     'max_leaf_nodes': [4, 6, 8],
# }, n_iter=6, random_state=0))

# I ran the code above and got the hyper-parameters below.
# I am hardcoding it now so it is explicit here.
# I also had to wrap the parameters in lists, and it seems that there is a
# missconfiguration in initial_height
hyper_params = [
    {
        "max_leaf_nodes": [4],
        # 'initial_height': [None],
        "basic_primitive": ["log,sqrt,sin,tanh"],
    },
    {
        "max_leaf_nodes": [8],
        # 'initial_height': ['0-6'],
        "basic_primitive": ["log,sqrt,sin,tanh"],
    },
    {
        "max_leaf_nodes": [6],
        # 'initial_height': ['0-6'],
        "basic_primitive": ["optimal"],
    },
    {
        "max_leaf_nodes": [6],
        # 'initial_height': ['0-6'],
        "basic_primitive": ["log,sqrt,sin,tanh"],
    },
    {
        "max_leaf_nodes": [8],
        # 'initial_height': [None],
        "basic_primitive": ["optimal"],
    },
    {
        "max_leaf_nodes": [8],
        # 'initial_height': [None],
        "basic_primitive": ["log,sqrt,sin,tanh"],
    },
]

# for g in hyper_params:
#     g['select'] = [selRandom] if g['initial_height'] is None else [selTournamentDCD]

est = PSTreeRegressor(
    regr_class=GPRegressor,
    tree_class=DecisionTreeRegressor,
    height_limit=6,
    n_pop=25,
    n_gen=1_000,
    normalize=True,
    basic_primitive="optimal",
    select=selTournamentDCD,
    size_objective=False,
    afp=True,
)


def complexity(est):
    _, _, total_complexity, _ = tree_gp_regressor_complexity(est)
    return int(total_complexity)


def model(e, X=None):
    return e.model()


eval_kwargs: dict[str, Any] = {}
