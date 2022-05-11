from dso import ParallelizedUnifiedDeepSymbolicRegressor

est = ParallelizedUnifiedDeepSymbolicRegressor()

def model(est, X=None):

    expr = est.expr

    if X is None or not hasattr(X, 'columns'):
        return expr

    features = X.columns

    for i in reversed(range(len(features))):
        before = "x{}".format(i + 1)
        after = "INPUT_{}_TUPNI".format(i)
        expr = expr.replace(before, after)

    for i, feature in enumerate(features):
        before = "INPUT_{}_TUPNI".format(i)
        after = feature
        expr = expr.replace(before, after)

    return expr

eval_kwargs = {}
