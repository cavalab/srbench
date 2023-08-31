from DAG_search.dag_search import DAGRegressor


est = DAGRegressor()

def model(est, X = None):
    # est.model() is a sympy expression
    if X is None:
        return str(est.model())
    else:
        # assuming X is pandas dataframe
        mapping = {'x_'+str(i):k for i,k in enumerate(X.columns)}
        new_model = str(est.model())
        for k,v in reversed(mapping.items()):
            new_model = new_model.replace(k,v)
        return new_model


eval_kwargs = dict(use_dataframe=False)