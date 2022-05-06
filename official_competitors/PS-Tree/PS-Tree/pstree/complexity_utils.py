import numpy as np

from .cluster_gp_sklearn import PseudoPartition


def tree_gp_regressor_complexity(regr):
    if not hasattr(regr.tree, 'tree_'):
        return [], 0, 0, {'number_of_features': 0}
    # complexity induced by non-leaf nodes
    split_node_count = regr.tree.tree_.node_count - regr.tree.tree_.n_leaves
    total_complexity = split_node_count

    # if there are some synthesized features used by non-leaf node, it should be considered
    coefs = np.max(np.abs(np.array([p['Ridge'].coef_ for p in regr.regr.pipelines])), axis=0)
    if regr.regr.adaptive_tree:
        if regr.regr.original_features == 'original' or isinstance(regr.tree, PseudoPartition):
            pass
        elif regr.regr.original_features:
            coefs += regr.tree.feature_importances_[regr.train_data.shape[1]:]
        else:
            coefs += regr.tree.feature_importances_

    # count nodes of symbolic models
    for i, g in enumerate(regr.regr.best_pop):
        if coefs[i] == 0:
            continue
        total_complexity += len(g)

    # complexity induced by coefficients of linear models
    total_complexity += np.count_nonzero(np.array([p['Ridge'].coef_ for p in regr.regr.pipelines]))
    total_complexity += np.count_nonzero(np.array([p['Ridge'].intercept_ for p in regr.regr.pipelines]))

    gene_list = []
    other_information = {
        'number_of_features': np.count_nonzero(coefs)
    }
    return gene_list, split_node_count, total_complexity, other_information
