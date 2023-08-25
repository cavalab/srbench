from DAG_search.dag_search import DAGRegressor


# probably more = better
hyper_params = [{'n_calc_nodes' : i} for i in [1, 2, 3, 4, 5]]

est = DAGRegressor()

def complexity(est):
    return est.complexity()

def model(est):
    # est.model() is a sympy expression
    return str(est.model())