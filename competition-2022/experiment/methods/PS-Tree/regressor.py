from sklearn.tree import DecisionTreeRegressor

from pstree.cluster_gp_sklearn import selTournamentDCD
from pstree.cluster_gp_sklearn import GPRegressor, PSTreeRegressor
from pstree.complexity_utils import tree_gp_regressor_complexity

hyper_params = [
    {}
]

est = PSTreeRegressor(regr_class=GPRegressor, tree_class=DecisionTreeRegressor,
                      height_limit=6, n_pop=25, n_gen=500,
                      normalize=True, basic_primitive='optimal',
                      select=selTournamentDCD, size_objective=False, afp=True)


def complexity(est):
    _, _, total_complexity, _ = tree_gp_regressor_complexity(est)
    return total_complexity


def model(e, X):
    """WGL: updated because model was returning wrong feature names"""
    model_str =  e.model()
    new_model = model_str
    mapping = {'X'+str(i): k for i, k in enumerate(X.columns)}
    for k, v in reversed(mapping.items()):
        new_model = new_model.replace(k, v)

    return new_model


eval_kwargs = {}
