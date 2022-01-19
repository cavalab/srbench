from pstree.cluster_gp_sklearn import GPRegressor, PSTreeRegressor, selTournamentDCD
from pstree.complexity_utils import tree_gp_regressor_complexity
from sklearn.tree import DecisionTreeRegressor

hyper_params = [
    {
    },
]

est = PSTreeRegressor(regr_class=GPRegressor, tree_class=DecisionTreeRegressor,
                      height_limit=6, n_pop=25, n_gen=500,
                      basic_primitive=False, size_objective=True,
                      select=selTournamentDCD)


def complexity(est):
    _, _, total_complexity, _ = tree_gp_regressor_complexity(est)
    return total_complexity


def model(e):
    return e.model()
