"""GP_ELITE method wrapper for SRBench.

Exposes the three objects the harness reads: `est`, `hyper_params`, `eval_kwargs`,
plus `model(est, X)` and `complexity(est)`.
"""
import time

import numpy as np

try:
    from sklearn.utils.validation import validate_data
except Exception:                       # older sklearn
    validate_data = None

from gp_elite import GPEliteRegressor
from gp_elite.api import symbolic_regression


class GPEliteSRBench(GPEliteRegressor):
    """GP_ELITE for SRBench.

    fit() runs sequential restarts (each one a full independent evolution) and
    keeps the best by validation R2, stopping once the time budget is nearly
    spent or `restarts` is reached.
    """

    def __init__(self, operators="physical", normalize="auto", generations=60,
                 speed="fast", validation_split=0.20, max_time=3600,
                 restarts=10, robust=False, parallel=None, random_state=0):
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
                # Publish the champion AS SOON AS it is found, so that a
                # single overrunning restart (symbolic_regression is not itself
                # time-bounded) can never leave the estimator without model_.
                # If the harness SIGALRM fires mid-restart, evaluate_model
                # still finds a (possibly worse) result instead of raising
                # AttributeError and losing the run entirely.
                self.model_ = best
                self.equation_ = best.expression
                self.is_fitted_ = True

            elapsed = time.time() - t0
            if elapsed > 0.85 * budget or elapsed + elapsed / (k + 1) > budget:
                break

        # Guarantee the attributes exist even if the very first restart failed
        # to produce a usable result (defensive; best is normally set above).
        if best is not None and not getattr(self, "is_fitted_", False):
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


# Map every GP_ELITE operator to a sympy-parsable form. Anything produced by the
# "physical" or "full" operator sets must appear here; unknown names would parse
# as opaque undefined functions (e.g. cube(x) is NOT x**3 to sympy) and break
# numeric evaluation downstream.
_UNARY = {
    "neg":     lambda a: "(-(%s))" % a,
    "abs":     lambda a: "Abs(%s)" % a,
    "inv":     lambda a: "(1/(%s))" % a,
    "sq":      lambda a: "((%s)**2)" % a,
    "cube":    lambda a: "((%s)**3)" % a,
    "sqrt":    lambda a: "sqrt(Abs(%s))" % a,   # engine's sqrt is domain-guarded
    "log":     lambda a: "log(Abs(%s))" % a,    # engine's log is domain-guarded
    "exp":     lambda a: "exp(%s)" % a,
    "sin":     lambda a: "sin(%s)" % a,
    "cos":     lambda a: "cos(%s)" % a,
    "tan":     lambda a: "tan(%s)" % a,
    "tanh":    lambda a: "tanh(%s)" % a,
    "step":    lambda a: "Heaviside(%s)" % a,
    "is_even": lambda a: "(1 - Mod(floor(%s), 2))" % a,
}
_BINARY = {
    "+":    lambda a, b: "((%s) + (%s))" % (a, b),
    "-":    lambda a, b: "((%s) - (%s))" % (a, b),
    "*":    lambda a, b: "((%s) * (%s))" % (a, b),
    "/":    lambda a, b: "((%s) / (%s))" % (a, b),
    "pow":  lambda a, b: "((%s)**(%s))" % (a, b),
    "max2": lambda a, b: "Max((%s), (%s))" % (a, b),
    "min2": lambda a, b: "Min((%s), (%s))" % (a, b),
}


def _to_sympy(node, cols):
    """Walk the expression tree, emit a sympy-parsable string with the
    dataset's real column names."""
    if node.left is None and node.right is None:
        v = node.value
        # np.float64 subclasses float, but np.int64 does NOT subclass int, so
        # test the numpy hierarchy explicitly — otherwise an integer-valued
        # numpy constant would fall through to the name branch and be emitted
        # as a feature name instead of a number.
        if isinstance(v, (int, float, np.integer, np.floating)) \
                and not isinstance(v, bool):
            return "%.12g" % float(v)
        s = str(v)                       # like 'X[3]' or 'X3'
        digits = "".join(c for c in s if c.isdigit())
        i = int(digits) if digits else 0
        return cols[i] if i < len(cols) else "x%d" % i

    op = node.value
    if node.right is None:               # unary
        a = _to_sympy(node.left, cols)
        f = _UNARY.get(op)
        if f is None:
            raise ValueError("unmapped unary operator %r in _to_sympy" % op)
        return f(a)

    a = _to_sympy(node.left, cols)
    b = _to_sympy(node.right, cols)
    f = _BINARY.get(op)
    if f is None:
        raise ValueError("unmapped binary operator %r in _to_sympy" % op)
    return f(a, b)


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
