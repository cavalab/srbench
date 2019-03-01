import numpy as np

def complexity(est):
    """Returns a complexity estimate for a model."""
    est_name = type(est).__name__

    if 'Feat' in est_name:
        # get_dim() here accounts for weights in umbrella ML model
        model_size = est.get_n_nodes()
    elif 'MLP' in est_name:
        model_size = np.sum([c.size for c in est.coefs_]+
                            [c.size for c in est.intercepts_])
    elif 'Torch' in est_name:
        model_size = est.module.get_n_params()
    elif hasattr(est,'coef_'):
        model_size = est.coef_.size
    elif 'RF' in est_name:
        model_size = np.sum([e.tree_.node_count for e in est.estimators_])
    elif 'XGB' in est_name:
        model_size = np.sum([m.count(':') for m in est._Booster.get_dump()])
    else:
        model_size = features.shape[1]

    return model_size
