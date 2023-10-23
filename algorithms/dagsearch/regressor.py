from DAG_search.dag_search import DAGRegressor
import multiprocessing

est = DAGRegressor(processes = 1) # increase number of processes here for speed up

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


eval_kwargs = {
    "use_dataframe": False,
    "test_params": {
        'n_calc_nodes' : 2,
        'max_orders' : 1000,
        'processes' : 1,
        'max_samples' : 100
    }
}