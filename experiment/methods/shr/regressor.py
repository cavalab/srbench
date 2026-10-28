# -*- coding: utf-8 -*-
"""SHR estimator interface for SRBench (experiment/evaluate_model.py).

Location: srbench_repo/experiment/methods/shr/regressor.py
Usage:
    python evaluate_model.py <data.tsv.gz> -ml shr -results_path results -seed 23654

Interface (required by the harness):
  - the estimator is a sklearn BaseEstimator/RegressorMixin with fit(X, y),
    predict(X), get_params();
  - model(est, X=None) returns the symbolic model string (variable names match
    the DataFrame columns of X);
  - est is a module-level instance; eval_kwargs holds optional evaluation config;
  - max_time is exposed: the harness sets it from fit_time_limit, and the engine
    uses it as a global per-problem budget, stopping new costly paths on timeout
    and finalizing the current best candidate.

Notes:
  - scale_x=False / scale_y=False: the method trains directly on raw, physical
    units (75% train). Structure recognition relies on the raw units; the
    harness allows scaling to be disabled via eval_kwargs (as AIFeynman does).
  - fit only receives the train split; the engine's internal fit/val uses a
    subset of train, and the test split is never touched.
  - the final test R2 is computed by the harness on the raw y.
"""

import os
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


def _load_engine():
    """Prefer a direct import (the install script adds the engine dir to
    sys.path); for local runs fall back to SHR_ENGINE / a relative path."""
    try:
        import shr_v7_full as SF  # after official install

        return SF
    except Exception:
        pass
    env = os.environ.get("SHR_ENGINE")
    cands = []
    if env:
        cands.append(env)
    here = os.path.dirname(os.path.abspath(__file__))
    # experiment/methods/shr/ -> local engine (development)
    cands.append(os.path.normpath(os.path.join(here, "..", "..", "..", "engine")))
    cands.append(os.path.normpath(os.path.join(here, "..", "..", "..")))
    for d in cands:
        if d and os.path.isdir(d) and os.path.isfile(os.path.join(d, "shr_v7_full.py")):
            if d not in sys.path:
                sys.path.insert(0, d)
            import shr_v7_full as SF  # noqa: E402

            return SF
    raise ImportError(
        "Cannot find the SHR engine: make sure the install script has set up "
        "shr_engine, or set SHR_ENGINE to the engine directory"
    )


SF = _load_engine()


class SHRv7Estimator(BaseEstimator, RegressorMixin):
    """SHR estimator wrapping deterministic solving and a GP fallback."""

    engine_: Any
    symbolic_model: Any

    def __init__(self, random_state=None, quick=False, max_time=None):
        self.random_state = random_state
        self.quick = quick
        self.max_time = max_time
        self.engine_ = None
        self.symbolic_model = None

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            names = [str(c) for c in X.columns]
            Xv = X.values
        else:
            Xv = np.asarray(X, float)
            names = ["x%d" % (i + 1) for i in range(Xv.shape[1])]
        yv = np.asarray(y, float).flatten()
        seed = 0 if self.random_state is None else int(self.random_state)
        self.engine_ = SF.fit_engine_full(
            Xv, yv, names, seed=seed, quick=self.quick, max_time=self.max_time
        )
        self.symbolic_model = self._to_string()
        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.asarray(X, float)
        pred = np.asarray(self.engine_.predict(X), float)
        if pred.ndim == 0:
            pred = np.full(X.shape[0], float(pred))
        pred = np.where(
            np.isfinite(pred),
            pred,
            np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0),
        )
        return pred

    def _to_string(self):
        m = self.engine_
        if m._kind == "gp":
            try:
                s = m.get_gp_string()
                if s:
                    return str(s)
            except Exception:
                pass
        # exact path: internal tuple -> simplified sympy expression
        try:
            import symbol_simplify as SS

            expr = SS.clean_sympy(m._expr, list(m.names0))
            if expr is not None:
                return str(expr)
        except Exception:
            # structure_guided nodes use 'c' leaves; convert to 'var' first
            try:
                import structure_guided as SG
                import symbol_simplify as SS

                expr = SS.clean_sympy(SG.c2var(m._expr), list(m.names0))
                if expr is not None:
                    return str(expr)
            except Exception:
                pass
        return str(m._expr)


def model(est, X=None):
    """Symbolic model hook; returns the expression over the original features."""
    return est.symbolic_model


def complexity(est):
    """Optional custom complexity (the harness defaults to sympy; this is the
    fallback if that fails)."""
    from sympy import count_ops, sympify

    try:
        expr = sympify(est.symbolic_model)
        n_atoms = len(expr.free_symbols)
        return int(count_ops(expr)) + n_atoms
    except Exception:
        return -1


est = SHRv7Estimator()


# Evaluation config: disable scaling (the method relies on raw units) and keep
# a DataFrame so physical names are passed. test_params only applies in the
# harness -test smoke mode (quick=True); the full run uses the default mode.
eval_kwargs = {
    "scale_x": False,
    "scale_y": False,
    "use_dataframe": True,
    "test_params": {"quick": True},
}
