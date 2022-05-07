from dso import ParallelUnifiedDeepSymbolicRegressor

est = ParallelUnifiedDeepSymbolicRegressor()

def model(est, X=None):

    expr = est.expr
    features = X.columns

    for i in range(len(features)):
        before = "x{}".format(i + 1)
        after = "INPUT_{}_TUPNI".format(i)
        expr = expr.replace(before, after)

    for i, feature in enumerate(features):
        before = "INPUT_{}_TUPNI".format(i)
        after = feature
        expr = expr.replace(before, after)

    return expr

eval_kwargs = {}
