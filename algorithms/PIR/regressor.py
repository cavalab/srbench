"""
SRBench submission: PIR (Physics Intermediate Representation) -- classical engine.

This file is GLUE only. It wraps the public, pip-installable classical PIR engine
(installed via install.sh from a stable source repo) in the interface SRBench
expects:

    est                 a scikit-learn-compatible Regressor instance
    model(est, X=None)  returns a sympy-parseable string for the fitted model
    eval_kwargs         method-specific args forwarded to evaluate_model.py

No JEPA, no flow prior, no OT loss. Vanilla classical PIR only -- this is the
configuration that produced the archived blind Tier A baseline (13/44 stable,
~29.5% mean over 5 seeds, results_tierA_blind/).

=============================================================================
THREE THINGS YOU MUST CONFIRM BEFORE OPENING THE PR  (do not skip these)
-----------------------------------------------------------------------------
[1] PUBLIC ENGINE URL. SRBench CI cannot pull a private repo. Set the repo
    URL + ref in install.sh. The import below must resolve from that public
    package.

[2] CONFIG MUST MATCH THE ARCHIVED BASELINE. The kwargs in _PIR_VANILLA_CONFIG
    below must be byte-for-byte the configuration used by sweep_tierA_blind.py
    that produced results_tierA_blind/. Otherwise the submitted number won't be
    the number you archived -- an integrity gap. Open sweep_tierA_blind.py and
    copy the exact PIRRegressor(...) kwargs here. The values below are
    placeholders based on the handoff (use_ot_loss=False, max_train_rows=800)
    and are NOT yet verified against the sweep script.

[3] VARIABLE NAMES. model() must return a string whose symbols match the
    SRBench dataset's column names (X.columns), NOT your local feynman_loader
    names. SRBench/PMLB Feynman columns may differ from ~/feynman_data/. Run the
    smoke test in SUBMISSION_NOTES.md against one real SRBench dataset and
    confirm the returned symbols match before trusting CI.
=============================================================================
"""

import signal
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

# The classical engine. install.sh pip-installs this from the PUBLIC repo.
# Path per handoff: physics_engine/sklearn_adapter.py defines PIRRegressor.
from physics_engine.sklearn_adapter import PIRRegressor


# --- [2] EXACT vanilla config that produced results_tierA_blind/ -------------
# CONFIRM these against sweep_tierA_blind.py before submitting. Anything that
# changes the discovered expression (filtering flags, search depth, subsample,
# seeds handling) belongs here so the submission reproduces the archived 13/44.
_PIR_VANILLA_CONFIG = dict(
    enforce_dimensions=False,        # blind sweep ran with dim-filter OFF
    allowed_powers=[1, 2],           # powers 1,2 only (the structural cap)
    include_pairwise_products=True,  # pairwise on; no 3-var assembly
    use_ransac=True,
    use_residual=True,
    use_sparse=True,
    use_ot_loss=False,
    add_physics_features=False,
    # random_state passed via the wrapper (defaults to 0, matching the
    # archived blind sweep which used SEED = 0). All other params
    # (alpha, beta, max_basis_terms, lambda_penalty, max_train_rows) are
    # left at adapter defaults -- the blind sweep did not set them.
)


class _Timeout(Exception):
    pass


def _on_alarm(signum, frame):
    raise _Timeout()


class PIRClassicRegressor(BaseEstimator, RegressorMixin):
    """Thin sklearn wrapper around classical PIRRegressor.

    Adds: a `max_time` budget enforced via SIGALRM (required by SRBench), a
    `random_state` attribute (required by SRBench), and a guaranteed-valid
    fallback model so model() never raises even if fit is interrupted.
    """

    def __init__(self, max_time=3600, random_state=0, **pir_kwargs):
        self.max_time = max_time
        self.random_state = random_state
        # Merge pinned vanilla config with any overrides passed by the harness.
        self.pir_kwargs = {**_PIR_VANILLA_CONFIG, **pir_kwargs}

    def _build(self):
        kw = dict(self.pir_kwargs)
        # Pass random_state through only if the engine accepts it; harmless
        # to set as attribute either way.
        try:
            return PIRRegressor(random_state=self.random_state, **kw)
        except TypeError:
            return PIRRegressor(**kw)

    def fit(self, X, y):
        # Guarantee a valid model exists before we risk a timeout: constant
        # = mean(y). evaluate_model.py can always score this.
        y_arr = np.asarray(y, dtype=float).ravel()
        self._fallback_expr_ = repr(float(np.mean(y_arr))) if y_arr.size else "0.0"
        self.expr_ = self._fallback_expr_
        self._inner = self._build()

        use_alarm = hasattr(signal, "SIGALRM") and self.max_time and self.max_time > 0
        old_handler = None
        if use_alarm:
            old_handler = signal.signal(signal.SIGALRM, _on_alarm)
            signal.alarm(int(self.max_time))
        try:
            self._inner.fit(X, y)
            # Adapter exposes model() -> str(self.expr_); fall back to .expr_.
            if hasattr(self._inner, "model"):
                self.expr_ = self._inner.model()
            elif hasattr(self._inner, "expr_"):
                self.expr_ = str(self._inner.expr_)
        except _Timeout:
            # Keep the constant fallback already stored in self.expr_.
            pass
        finally:
            if use_alarm:
                signal.alarm(0)
                if old_handler is not None:
                    signal.signal(signal.SIGALRM, old_handler)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        if hasattr(self, "_inner") and hasattr(self._inner, "predict"):
            try:
                return self._inner.predict(X)
            except Exception:
                pass
        # Fallback: constant prediction (matches the fallback expr).
        n = X.shape[0] if hasattr(X, "shape") else len(X)
        try:
            val = float(self._fallback_expr_)
        except (TypeError, ValueError):
            val = 0.0
        return np.full(n, val, dtype=float)

    def model(self):
        return self.expr_


# The estimator SRBench will fit. max_time is the param SRBench controls;
# the harness also sends SIGALRM at the process level if fit() overruns.
est = PIRClassicRegressor(max_time=3600, random_state=None)


def model(est, X=None):
    """Return a sympy-parseable model string with symbols matching X.columns.

    Uses the SRBench-documented mapping idiom only if the engine emitted
    generic names (x_0, x0, X0, ...). If the engine already uses the dataset
    column names (as the handoff example '1.0*Ef*q2' suggests), the string is
    returned unchanged.
    """
    expr = est.model() if hasattr(est, "model") else str(getattr(est, "expr_", "0.0"))

    if X is None or not hasattr(X, "columns"):
        return expr

    cols = list(X.columns)
    # If any real column name already appears, assume names are correct.
    if any(str(c) in expr for c in cols):
        return expr

    # Otherwise remap generic positional names -> dataset columns.
    # reversed() so 'x_1' doesn't clobber the prefix of 'x_10'.
    import re
    for prefix in ("x_", "x", "X_", "X"):
        if re.search(rf"\b{prefix}\d+\b", expr):
            mapping = {f"{prefix}{i}": str(k) for i, k in enumerate(cols)}
            for k, v in reversed(list(mapping.items())):
                expr = re.sub(rf"\b{re.escape(k)}\b", v, expr)
            break
    return expr


# --- forwarded to evaluate_model.py ------------------------------------------
# CRITICAL: scale_x/scale_y MUST be False. SRBench StandardScales X and y by
# default, which destroys the units and exact coefficients PIR depends on
# (a scaled run recovers ~nothing). The dev harness already forces these off on
# the symbolic-data track, but we pin them so a non-sym run can't quietly zero
# us out. skip_tuning=True: PIR has no GridSearch tuning step to run.
#
# NOTE: these keys target the `dev` branch signature (where PRs go):
#   evaluate_model(..., scale_x, scale_y, pre_train, skip_tuning, sym_data)
# If you test against `master` instead, it uses `test_params` rather than
# `skip_tuning`; see SUBMISSION_NOTES.md.
eval_kwargs = {
    "scale_x": False,
    "scale_y": False,
    "skip_tuning": True,
}
