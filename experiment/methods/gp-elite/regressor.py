"""SRBench algorithm definition for GP_ELITE.

GP_ELITE: pure-Python symbolic regression (genetic programming with
Levenberg-Marquardt constant optimization, multi-restart, Pareto output).
Package: https://pypi.org/project/gp-elite/  (pip install gp-elite)
"""
import time

import numpy as np
from gp_elite import GPEliteRegressor
from gp_elite.api import symbolic_regression

try:
    from sklearn.utils.validation import validate_data
except Exception:
    validate_data = None


class GPEliteSRBench(GPEliteRegressor):
    """GPEliteRegressor + the max_time contract required by SRBench.

    fit() runs sequential restarts (each one a full independent evolution)
    and keeps the best model by internal validation R2, stopping when the
    time budget is nearly spent or `restarts` is reached.
    """

    def __init__(self, operators="physical", normalize="auto",
                 generations=60, speed="fast", validation_split=0.20,
                 restarts=10, robust=False, parallel=None, random_state=0,
                 max_time=3600):
        super().__init__(operators=operators, normalize=normalize,
                         generations=generations, speed=speed,
                         validation_split=validation_split, restarts=restarts,
                         robust=robust, parallel=parallel,
                         random_state=random_state)
        self.max_time = max_time

    def fit(self, X, y):
        if validate_data is not None:
            X, y = validate_data(self, X, y, y_numeric=True,
                                 ensure_min_samples=2, dtype="numeric")
        else:
            X = np.asarray(X, dtype=float)
            y = np.asarray(y, dtype=float)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            self.n_features_in_ = X.shape[1]
        y = np.ravel(y)

        names = ["X%d" % i for i in range(self.n_features_in_)]
        t0 = time.time()
        budget = max(30.0, float(self.max_time))
        best, best_score = None, -np.inf
        seed0 = 0 if self.random_state is None else int(self.random_state)
        for k in range(max(1, int(self.restarts))):
            r = symbolic_regression(
                X, y, feature_names=names,
                operators=self.operators, normalize=self.normalize,
                generations=self.generations, speed=self.speed,
                validation_split=self.validation_split, restarts=1,
                robust=self.robust, parallel=self.parallel,
                seed=seed0 + 1000 * k)
            score = r.r2_validation
            if score is None or not np.isfinite(score):
                p = r.predict(X)
                ss = float(np.sum((y - y.mean()) ** 2)) or 1e-30
                score = 1.0 - float(np.sum((y - p) ** 2) / ss)
            if score > best_score:
                best, best_score = r, score
            elapsed = time.time() - t0
            if elapsed > 0.85 * budget or elapsed + elapsed / (k + 1) > budget:
                break
        self.model_ = best
        self.equation_ = best.expression
        self.is_fitted_ = True
        return self


# ── the three objects SRBench reads ─────────────────────────────────────────

est = GPEliteSRBench(operators="physical", generations=60, speed="fast",
                     restarts=10, max_time=3600, random_state=0)

hyper_params = [
    {"operators": ("physical", "full"),
     "generations": (30, 60)},
]

eval_kwargs = {}


def _to_sympy(node, cols):
    """Walk the expression tree, emit a sympy-parsable string with the
    dataset's real column names."""
    if node.left is None and node.right is None:
        v = node.value
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return "%.12g" % float(v)
        s = str(v)                       # like 'X[3]' or 'X3'
        digits = "".join(c for c in s if c.isdigit())
        i = int(digits) if digits else 0
        return cols[i] if i < len(cols) else "x%d" % i
    op = node.value
    if node.right is None:               # unary
        a = _to_sympy(node.left, cols)
        return {"sq":   "((%s)**2)" % a,
                "neg":  "(-(%s))" % a,
                "inv":  "(1/(%s))" % a,
                "abs":  "Abs(%s)" % a,
                }.get(op, "%s(%s)" % (op, a))
    a = _to_sympy(node.left, cols)
    b = _to_sympy(node.right, cols)
    if op == "pow":
        return "((%s)**(%s))" % (a, b)
    return "((%s) %s (%s))" % (a, op, b)


def model(est, X=None):
    cols = list(X.columns) if X is not None and hasattr(X, "columns") \
        else ["x%d" % i for i in range(getattr(est, "n_features_in_", 0))]
    # Fold the input normalization into the string so the formula, evaluated
    # on RAW columns, reproduces est.predict exactly. The engine's shift-free
    # scaler divides each feature by scale_[i].
    scaler = getattr(est.model_, "scaler", None)
    if scaler is not None and hasattr(scaler, "scale_"):
        cols = ["((%s)/%.12g)" % (c, float(f))
                for c, f in zip(cols, scaler.scale_)]
    return _to_sympy(est.model_.node, cols)


def complexity(est):
    return int(est.model_.size)

