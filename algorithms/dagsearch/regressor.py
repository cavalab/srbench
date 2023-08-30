from DAG_search.dag_search import DAGRegressor


est = DAGRegressor()

def model(est, X = None):
    # est.model() is a sympy expression
    return str(est.model())