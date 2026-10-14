"""GP_ELITE method wrapper for SRBench.

Exposes the three objects the harness reads: `est`, `hyper_params`, `eval_kwargs`,
plus `model(est, X)` and `complexity(est)`.

Requires gp-elite >= 0.8.0, which brings the two things this wrapper relies on:
a per-call time limit (`time_limit=`), so a restart can no longer overrun the
harness budget, and an exported formula written in the raw input variables and
checked against `predict` at fit time (`SRResult.sympy`).
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
    keeps the best by validation R2. Every restart is time-bounded: it receives
    an equal share of the budget still available, so the whole fit ends within
    `max_time` (minus a safety margin) and the champion is published as soon as
    it is found.
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

    def _publish(self, r):
        self.model_ = r
        self.equation_ = r.expression
        self.is_fitted_ = True

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
        # 10 % margin: final selection, predict and model export happen after
        # the last restart, and the harness alarm must never be the one that
        # ends the fit.
        deadline = t0 + 0.90 * budget
        n = max(1, int(self.restarts))
        best, best_score = None, -np.inf
        seed0 = 0 if self.random_state is None else int(self.random_state)

        for k in range(n):
            remaining = deadline - time.time()
            if k > 0 and remaining <= 0:
                break
            r = symbolic_regression(
                X, y, feature_names=names,
                operators=self.operators, normalize=self.normalize,
                generations=self.generations, speed=self.speed,
                validation_split=self.validation_split, restarts=1,
                robust=self.robust, parallel=self.parallel,
                time_limit=max(1.0, remaining / (n - k)),
                seed=seed0 + 1000 * k)
            score = r.r2_validation
            if score is None or not np.isfinite(score):
                p = r.predict(X)
                ss = float(np.sum((y - y.mean()) ** 2)) or 1e-30
                score = 1.0 - float(np.sum((y - p) ** 2) / ss)
            if best is None or score > best_score:
                best, best_score = r, score
                self._publish(best)          # published as soon as found

        if best is not None and not getattr(self, "is_fitted_", False):
            self._publish(best)
        return self


# ── the three objects SRBench reads ─────────────────────────────────────────

est = GPEliteSRBench(operators="physical", generations=60, speed="fast",
                     restarts=10, max_time=3600, random_state=0)

hyper_params = [
    {"operators": ("physical", "full"),
     "generations": (30, 60)},
]

eval_kwargs = {}


def model(est, X=None):
    """The fitted formula as a sympy-parsable string in the dataset's columns.

    gp-elite >= 0.8 writes it in the raw input variables for every internal
    normalisation (division by max|x| by default, min-max or z-score on
    request) and checks at fit time that it reproduces predict(); protected
    operators are spelled as the engine computes them (sqrt(Abs(u)) where u
    can be negative).
    """
    cols = list(X.columns) if X is not None and hasattr(X, "columns") \
        else ["x%d" % i for i in range(getattr(est, "n_features_in_", 0))]
    return est.model_.sympy(cols)


def complexity(est):
    return int(est.model_.size)
