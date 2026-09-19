"""
SRBench method: PIR (Physics Intermediate Representation) — classical engine.

Importable module for the SRBench harness (imported as `methods.pir`).
Wraps the public, pip-installable classical PIR engine (installed via
algorithms/pir/install.sh) in the interface SRBench expects:

    est                    a scikit-learn-compatible Regressor instance
    model(est, X=None)     returns a sympy-parseable string for the fitted model
    complexity(est)        returns an integer complexity count for the model
    eval_kwargs            method-specific args forwarded to evaluate_model.py

No neural components. Vanilla classical PIR only — this is the configuration
that produced the verified blind Tier A baseline:

    12/44 EXACT  (zero seed wobble, v3.4)
    +12/44 FORM_NUMERIC secondary (correct functional form, transcendental
                constant folded as decimal — reported separately, never summed)

Engine: https://github.com/Qazi-pk/physics-engine (MIT, tag v3.4.1)
Paper: https://doi.org/10.5281/zenodo.21351039
"""

import re

import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin

# The classical engine. algorithms/pir/install.sh pip-installs this from the
# public repo at tag v3.4.1.
from physics_engine.sklearn_adapter import PIRRegressor


# --- Vanilla config that produced the verified blind Tier A result ------------
_PIR_VANILLA_CONFIG = dict(
    enforce_dimensions=False,        # blind sweep ran with dim-filter OFF
    allowed_powers=[1, 2],           # powers 1,2 only (structural cap)
    include_pairwise_products=True,  # pairwise on; no 3-var assembly
    use_ransac=True,
    use_residual=True,
    use_sparse=True,
    use_ot_loss=False,
    add_physics_features=False,
)


class PIRClassicRegressor(BaseEstimator, RegressorMixin):
    """Thin sklearn wrapper around the classical PIRRegressor.

    Adds a `random_state` attribute and a guaranteed-valid fallback model so
    that model() never raises even if the inner fit fails. Timeout
    enforcement is left entirely to SRBench's own outer alarm: signal.alarm
    is process-global, so a second SIGALRM set here would silently overwrite
    SRBench's own timeout handler (per gAldeia's review on PR #210). The
    `max_time` constructor kwarg is still accepted, for API compatibility
    with the harness's test_params override, but is currently unused.
    """

    def __init__(self, max_time=3600, random_state=None, **pir_kwargs):
        self.max_time = max_time
        self.random_state = random_state
        self.pir_kwargs = {**_PIR_VANILLA_CONFIG, **pir_kwargs}

    def _build(self):
        kw = dict(self.pir_kwargs)
        seed = 0 if self.random_state is None else self.random_state
        try:
            return PIRRegressor(random_state=seed, **kw)
        except TypeError:
            # Engine may not accept random_state; harmless to omit.
            return PIRRegressor(**kw)

    def fit(self, X, y):
        # Guarantee a valid model exists before risking a timeout:
        # constant = mean(y). evaluate_model.py can always score this.
        y_arr = np.asarray(y, dtype=float).ravel()
        self._fallback_expr_ = repr(float(np.mean(y_arr))) if y_arr.size else "0.0"
        self.expr_ = self._fallback_expr_
        self._inner = self._build()

        # No process-level alarm here — SRBench's own outer timeout
        # enforces max_time; see class docstring.
        try:
            self._inner.fit(X, y)
            if hasattr(self._inner, "model"):
                self.expr_ = self._inner.model()
            elif hasattr(self._inner, "expr_"):
                self.expr_ = str(self._inner.expr_)
        except Exception:
            # Keep the constant fallback already stored in self.expr_.
            pass
        self.is_fitted_ = True
        return self

    def predict(self, X):
        if hasattr(self, "_inner") and hasattr(self._inner, "predict"):
            try:
                return self._inner.predict(X)
            except Exception:
                pass
        # Fallback: constant prediction matching the fallback expression.
        n = X.shape[0] if hasattr(X, "shape") else len(X)
        try:
            val = float(self._fallback_expr_)
        except (TypeError, ValueError):
            val = 0.0
        return np.full(n, val, dtype=float)

    def model(self):
        return self.expr_


# The estimator SRBench will fit.
est = PIRClassicRegressor(max_time=3600, random_state=None)


def model(est, X=None):
    """Return a sympy-parseable model string with symbols matching X.columns.

    If the engine already emits dataset column names, the string is returned
    unchanged. Otherwise generic positional names (x_0, x0, X0, ...) are
    remapped to the dataset columns.
    """
    expr = (
        est.model()
        if hasattr(est, "model")
        else str(getattr(est, "expr_", "0.0"))
    )

    if X is None or not hasattr(X, "columns"):
        return expr

    cols = list(X.columns)
    # Word-boundary match, not substring: a column literally named "co"
    # must not match inside "cos(x_0)" (per gAldeia's review on PR #210).
    def _col_appears(col_name, s):
        return re.search(r"\b" + re.escape(str(col_name)) + r"\b", s) is not None
    # If any real column name already appears, assume names are correct.
    if any(_col_appears(c, expr) for c in cols):
        return expr

    # Remap generic positional names -> dataset columns.
    # reversed() so 'x_1' doesn't clobber the prefix of 'x_10'.
    for prefix in ("x_", "x", "X_", "X"):
        if re.search(rf"\b{prefix}\d+\b", expr):
            mapping = {f"{prefix}{i}": str(k) for i, k in enumerate(cols)}
            for k, v in reversed(list(mapping.items())):
                expr = re.sub(rf"\b{re.escape(k)}\b", v, expr)
            break
    return expr


def complexity(est):
    """Integer complexity of the fitted model.

    Counts nodes in the expression by splitting on operators and separators,
    following the convention used by other SRBench methods (e.g. gplearn).
    SRBench normally uses SymPy's own complexity first; this is only a
    fallback, but the operator count is included so the fallback is as
    accurate as possible (per gAldeia's review on PR #210).
    """
    expr = model(est)
    if not expr:
        return 0
    # Count operand-like tokens and add back the operators the split removed.
    tokens = re.split(r"[\s\(\),\+\-\*\/\^]+", expr)
    operand_count = len([t for t in tokens if t])
    operator_count = len(re.findall(r"[\+\-\*\/\^]", expr))
    return operand_count + operator_count


# --- forwarded to evaluate_model.py ------------------------------------------
# CRITICAL: scale_x/scale_y MUST be False. SRBench StandardScales X and y by
# default, which destroys the units and exact coefficients PIR depends on.
# `test_params` shortens run-time during CI smoke tests (master-branch signature).
eval_kwargs = {
    "scale_x": False,
    "scale_y": False,
    "test_params": {"max_time": 60},
}
