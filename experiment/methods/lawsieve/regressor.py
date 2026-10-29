"""SRBench wrapper for LawSieve (https://github.com/Ephemeral6/lawsieve).

LawSieve is a deterministic, pure-Python symbolic regression method for noise-robust
recovery of closed-form laws (best-first recursive reduction with raw-space refits).
algorithms/lawsieve/install.sh installs it from a pinned release tag.

Time budget: the harness sets `est.max_time`; fit() reads it at call time and stops
before it (it plans for ~0.95 * max_time - 3 s and emits its best model so far).
"""

from typing import Any

from lawsieve import LawSieveRegressor

est = LawSieveRegressor(max_time=60, random_state=None)

# Deterministic given random_state (which seeds the internal search/holdout split);
# a single empty config, i.e. no hyperparameter tuning.
hyper_params: list[dict[str, Any]] = [{}]

# Raw features and target: the reductions use the physical (positive) variable ranges.
eval_kwargs = {"scale_x": False, "scale_y": False, "use_dataframe": False}


def model(est, X=None):
    """Sympy-compatible string of the fitted model in the dataset's feature names."""
    if X is not None and hasattr(X, "columns"):
        names = [str(c) for c in X.columns]
    else:
        names = [f"x_{i}" for i in range(est.n_features_in_)]
    return est.model_string(names)


def complexity(est):
    return int(est.expr_.size())
