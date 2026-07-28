"""SRBench entry for lagh (certified symbolic law discovery).

Contract objects: `est` (sklearn-compatible regressor), `model(est, X)`
(sympy-compatible string), `eval_kwargs`.
"""
from lagh.sklearn import LaghRegressor

est = LaghRegressor(max_time=3600)


def model(est, X=None):
    """Sympy-compatible model string, with x_i mapped to X's column names
    (SRBench requirement: variable names must match the training DataFrame)."""
    m = est.model()
    if X is not None and hasattr(X, "columns"):
        mapping = {"x_" + str(i): k for i, k in enumerate(X.columns)}
        for k, v in reversed(list(mapping.items())):
            m = m.replace(k, str(v))
    return m


eval_kwargs = {}
