def complexity(est):
    """Returns a complexity estimate for a model."""
    est_name = type(est).__name__

    if 'Feat' in est_name:
        # get_dim() here accounts for weights in umbrella ML model
        num_params = best_est.get_n_params()+best_est.get_dim()
        model_size = best_est.get_n_nodes()
    elif 'MLP' in est_name:
        num_params = np.sum([c.size for c in best_est.coefs_]+
                            [c.size for c in best_est.intercepts_])
    elif 'Torch' in est_name:
        num_params = best_est.module.get_n_params()
    elif hasattr(best_est,'coef_'):
        num_params = best_est.coef_.size
        model_size = num_params
    elif 'RF' in est_name:
        model_size = np.sum([e.tree_.node_count for e in best_est.estimators_])
    elif 'XGB' in est_name:
        model_size = np.sum([m.count(':') for m in best_est._Booster.get_dump()])
    else:
        model_size = features.shape[1]
        num_params = model_size
