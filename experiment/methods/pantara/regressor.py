"""
Pantara v7d — SRBench-compatible wrapper.

Format: experiment/methods/<algo>/regressor.py
Required exports: est, model(est, X=None)
Optional exports: eval_kwargs, hyper_params, complexity

Interface contract (from evaluate_model.py):
  est.fit(X_train, y_train)       — X may be a DataFrame or ndarray
  est.predict(X)                  — returns 1-D array
  model(est, X_df)                — returns sympy-compatible string or None
  est.max_time                    — set by harness before fit()
  est.random_state                — set by harness before fit()

Design notes:
  • eval_kwargs disables StandardScaler because Pantara requires positive
    X and y for log-log power-law detection.
  • fit() stores one "block recipe" per matching-pursuit step so that
    predict() can replay the model on held-out data without access to y.
  • Power-law blocks are exact (lstsq coefficients).
  • Trig blocks are exact (stored A, ω, φ, C).
  • Linear/polynomial blocks use OLS replay (equivalent to Adam's
    converged solution for the MSE objective).
  • When feature-selection cannot be reversed (rare edge cases), the
    block is silently dropped — predict() returns partial sum.
"""

import os
import sys
import numpy as np
import warnings

import torch
import pantara
from pantara import (
    OracleClassifier, N_CLASSES, DEVICE, FAMILIES, N_POINTS, MIN_SCORE,
    best_per_family_multispace, make_state_stats, make_oracle_input,
    fit_trig, detect_and_fit_power_law, bloc_score, power_law_bloc_score,
    transform_space, inverse_transform_y, corr_np, standardize,
)

_MODEL_PATH = pantara.MODEL_PATH

# ─────────────────────────────────────────────────────────────────────────────
# BLOCK RECIPES
# Each recipe is a dict with a 'type' key and enough state to call predict().
# ─────────────────────────────────────────────────────────────────────────────

def _predict_power_law(block, X):
    """y = C · x0^n0 · x1^n1 · ..."""
    eps = 1e-10
    coeffs = block['coeffs']          # (D+1,): [log C, n1, n2, ...]
    N = len(X)
    log_X = np.log(np.abs(X) + eps)
    A = np.column_stack([np.ones(N), log_X])
    pred_log = A @ coeffs
    return np.exp(pred_log)


def _predict_trig(block, X):
    """y = A · sin/cos(ω·xj + φ) + C   (or + B·xj for linear variant)"""
    j   = block['j_var']
    x   = X[:, j]
    A   = block['A']
    w   = block['omega']
    phi = block['phi']
    C   = block['C']
    form = block['form']
    if form == 'sin':
        return A * np.sin(w * x + phi) + C
    elif form == 'cos':
        return A * np.cos(w * x + phi) + C
    elif form == 'sin+lin':
        B = block.get('B', 0.0)
        return A * np.sin(w * x + phi) + B * x + C
    else:  # cos+lin
        B = block.get('B', 0.0)
        return A * np.cos(w * x + phi) + B * x + C


def _predict_1d_linear(block, X):
    """Replay a 1-D OLS block on new X."""
    space   = block['space']
    family  = block['family']
    indices = block['indices']         # tuple of feature indices
    mn_c    = block['mn_c']
    iqr_c   = block['iqr_c']
    mn_r    = block['mn_r']
    iqr_r   = block['iqr_r']
    w       = block['w']
    b       = block['b']

    X_t, _, valid = transform_space(X, np.zeros(len(X)), space)
    if not valid:
        return np.zeros(len(X))

    col_new = _compute_feature_col(family, indices, X_t)
    if col_new is None:
        return np.zeros(len(X))

    col_n   = (col_new - mn_c) / (iqr_c + 1e-10)
    pred_t  = (w * col_n + b) * iqr_r + mn_r
    return inverse_transform_y(pred_t, space)


def _predict_multi_linear(block, X):
    """Replay a multivariate OLS block (log-space) on new X."""
    space = block['space']
    mu_x  = block['mu_x']
    s_x   = block['s_x']
    mu_r  = block['mu_r']
    s_r   = block['s_r']
    W     = block['W']
    b     = block['b']

    if space == 'log_log_multi':
        eps  = 1e-10
        X_t  = np.log(np.abs(X) + eps)
    else:
        X_t, _, valid = transform_space(X, np.zeros(len(X)), space)
        if not valid:
            return np.zeros(len(X))

    # Pad or truncate to match stored dimension
    D_stored = len(mu_x)
    D_new    = X_t.shape[1]
    if D_new < D_stored:
        X_t = np.column_stack([X_t, np.zeros((len(X_t), D_stored - D_new))])
    elif D_new > D_stored:
        X_t = X_t[:, :D_stored]

    X_std  = (X_t - mu_x) / (s_x + 1e-10)
    pred_s = X_std @ W + b
    pred_t = pred_s * s_r + mu_r

    if space == 'log_log_multi':
        return np.exp(pred_t)   # Fix B from v7d
    return inverse_transform_y(pred_t, space)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _compute_feature_col(family, indices, X_t):
    j = indices[0]
    D = X_t.shape[1]
    if j >= D:
        return None
    if family == 'lin':
        return X_t[:, j].copy()
    elif family == 'sq':
        return X_t[:, j] ** 2
    elif family == 'pair':
        k = indices[1]
        if k >= D: return None
        return X_t[:, j] * X_t[:, k]
    elif family == 'qc':
        k = indices[1]
        if k >= D: return None
        return X_t[:, j] * X_t[:, k] ** 2
    elif family == 'trip':
        k, l = indices[1], indices[2]
        if k >= D or l >= D: return None
        return X_t[:, j] * X_t[:, k] * X_t[:, l]
    elif family == 'qinv':
        k = indices[1]
        if k >= D: return None
        return X_t[:, j] ** 2 / (X_t[:, k] + 1e-10)
    return None


def _find_feature_indices(family, col, X_t):
    """Reverse-engineer which feature combination produced `col` from X_t."""
    D = X_t.shape[1]

    def match(c):
        # Compare in standardized form (shape-invariant)
        c_s  = c - c.mean();  c_n  = c_s  / (np.std(c_s)  + 1e-10)
        co_s = col - col.mean(); co_n = co_s / (np.std(co_s) + 1e-10)
        return float(np.mean((c_n - co_n) ** 2)) < 1e-4

    if family == 'lin':
        for j in range(D):
            if match(X_t[:, j]):
                return (j,)
    elif family == 'sq':
        for j in range(D):
            if match(X_t[:, j] ** 2):
                return (j,)
    elif family == 'pair':
        for j in range(D):
            for k in range(j + 1, D):
                if match(X_t[:, j] * X_t[:, k]):
                    return (j, k)
    elif family == 'qc':
        for j in range(D):
            for k in range(D):
                if j != k and match(X_t[:, j] * X_t[:, k] ** 2):
                    return (j, k)
    elif family == 'trip':
        for j in range(D):
            for k in range(j + 1, D):
                for l in range(k + 1, D):
                    if match(X_t[:, j] * X_t[:, k] * X_t[:, l]):
                        return (j, k, l)
    elif family == 'qinv':
        for j in range(D):
            for k in range(D):
                if j != k and np.all(np.abs(X_t[:, k]) > 1e-10):
                    if match(X_t[:, j] ** 2 / X_t[:, k]):
                        return (j, k)
    return None


def _fit_ols_1d(col_t, resid_t):
    """OLS linear fit in normalized space. Returns (mn_c, iqr_c, mn_r, iqr_r, w, b)."""
    mn_c  = float(np.median(col_t))
    iqr_c = float(np.percentile(col_t, 75) - np.percentile(col_t, 25)) + 1e-10
    mn_r  = float(np.median(resid_t))
    iqr_r = float(np.percentile(resid_t, 75) - np.percentile(resid_t, 25)) + 1e-10

    col_n   = (col_t - mn_c) / iqr_c
    resid_n = (resid_t - mn_r) / iqr_r

    A = np.column_stack([col_n, np.ones(len(col_n))])
    params, *_ = np.linalg.lstsq(A, resid_n, rcond=None)
    w, b = float(params[0]), float(params[1])
    return mn_c, iqr_c, mn_r, iqr_r, w, b


def _fit_ols_multi(X_t, resid_t):
    """OLS multivariate fit. Returns (mu_x, s_x, mu_r, s_r, W, b)."""
    N, D   = X_t.shape
    mu_x   = X_t.mean(axis=0)
    s_x    = X_t.std(axis=0) + 1e-10
    X_std  = (X_t - mu_x) / s_x

    mu_r   = float(resid_t.mean())
    s_r    = float(resid_t.std()) + 1e-10
    resid_s = (resid_t - mu_r) / s_r

    A = np.column_stack([X_std, np.ones(N)])
    params, *_ = np.linalg.lstsq(A, resid_s, rcond=None)
    W = params[:-1]
    b = float(params[-1])
    return mu_x, s_x, mu_r, s_r, W, b


# ─────────────────────────────────────────────────────────────────────────────
# PANTARA REGRESSOR
# ─────────────────────────────────────────────────────────────────────────────

class PantaraRegressor:
    """
    Pantara v7d — Matching-pursuit symbolic regression.

    Fits a sum of basis functions chosen by a neural oracle, with:
    - Analytical power-law detection in log-log space (O(N) lstsq)
    - 8 function families: lin, sq, pair, qc, trip, qinv, sin, cos
    - 6 transformation spaces: original, log_y, log_x, log_log, sq_y, inv_x
    - 5 matching-pursuit steps maximum

    Attributes stored after fit():
    - fitted_blocks_  : list of block recipes for predict()
    - chosen_         : list of block names (for model() and diagnostics)
    - feature_names_  : list of column names (from DataFrame input)
    """

    def __init__(self, max_time=3600, random_state=42):
        self.max_time     = max_time    # set by SRBench harness
        self.random_state = random_state

    # ── helpers ──────────────────────────────────────────────────────────────

    def _load_oracle(self):
        oracle = OracleClassifier(n_classes=N_CLASSES)
        oracle.load_state_dict(torch.load(_MODEL_PATH, weights_only=True))
        oracle.to(DEVICE)
        oracle.eval()
        return oracle

    def _to_numpy(self, X):
        """Accept DataFrame or ndarray, return float64 ndarray (N, D)."""
        import pandas as pd
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            return X.values.astype(np.float64)
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if not hasattr(self, 'feature_names_') or self.feature_names_ is None:
            self.feature_names_ = [f'x{j}' for j in range(arr.shape[1])]
        return arr

    # ── fit ──────────────────────────────────────────────────────────────────

    def fit(self, X, y):
        np.random.seed(self.random_state)

        X_np  = self._to_numpy(X)
        y_np  = np.asarray(y, dtype=np.float64).ravel()

        self.fitted_blocks_ = []
        self.chosen_        = []

        oracle = self._load_oracle()
        resid  = y_np.copy()

        for _step in range(5):
            ratio = resid.std() / (y_np.std() + 1e-10)
            if ratio < 0.06:
                break

            scores, best_config, _ = best_per_family_multispace(X_np, resid, y_np)
            stats  = make_state_stats(scores, resid, y_np, X_np)
            points = make_oracle_input(X_np, resid, y_np, N_POINTS)
            ranked, probs = oracle.predict_ranked(points, stats)

            # ── power law first (v7d logic) ──────────────────────────────
            if np.all(X_np > 0) and np.all(resid > 0):
                pl = detect_and_fit_power_law(X_np, resid)
                if pl is not None:
                    pred_pl, coeffs_pl, _ = pl
                    new_resid = resid - pred_pl
                    quality   = power_law_bloc_score(resid, new_resid, y_np)
                    if quality > 0.30:
                        self.fitted_blocks_.append({
                            'type': 'power_law',
                            'coeffs': coeffs_pl.copy(),
                        })
                        self.chosen_.append('power_law[log_log_lstsq]')
                        resid = new_resid
                        if quality > 0.80:
                            break
                        continue

            # ── oracle-guided candidate selection ────────────────────────
            actions = list(ranked[:4])

            # bonus lin[original]
            lin_idx = FAMILIES.index('lin')
            sc_l, sp_l, _, _, _ = best_config['lin']
            if sp_l == 'original' and sc_l >= 0.85 and lin_idx not in actions:
                actions.append(lin_idx)

            # bonus pair[original]
            pair_idx = FAMILIES.index('pair')
            sc_p, sp_p, _, _, _ = best_config['pair']
            if sp_p == 'original' and sc_p >= 0.85 and pair_idx not in actions:
                actions.append(pair_idx)

            candidates = []
            for action in actions:
                if action >= len(FAMILIES):
                    continue
                fam = FAMILIES[action]
                sc, sp, col, X_t, resid_t = best_config[fam]
                if col is None:
                    continue

                # ── trig: use specialized estimator ─────────────────────
                if fam in ('sin', 'cos'):
                    try:
                        omega_init = float(sp.split('_w')[-1])
                    except Exception:
                        omega_init = None
                    try:
                        j_var = int(sp.split('_x')[1].split('_')[0])
                    except Exception:
                        j_var = 0
                    new_resid, pred_orig, trig_info = fit_trig(
                        X_np[:, j_var], resid, y_np,
                        omega_candidates=[omega_init] if omega_init else None,
                        try_linear=False,
                        steps=600,
                    )
                    sc_after, _, _ = best_per_family_multispace(X_np, new_resid, y_np)
                    quality  = bloc_score(resid, new_resid, y_np, sc_after)
                    clf_w    = float(probs[action]) / (float(probs[ranked[0]]) + 1e-10)
                    combined = quality * (0.6 + 0.4 * clf_w)
                    candidates.append({
                        'family': fam, 'space': sp, 'action': action,
                        'quality': quality, 'combined': combined,
                        'new_resid': new_resid, 'pred': pred_orig,
                        'trig_info': trig_info,
                        'j_var': j_var,
                    })
                    continue

                # ── non-trig: run original pipeline step ─────────────────
                # Determine if multifeature applies
                use_multi = (
                    sp in ('log_log', 'log_x', 'log_y', 'log_log_multi')
                    and X_t is not None
                    and X_t.ndim == 2
                    and X_t.shape[1] > 1
                )

                # Reproduce prediction (same as fit_block in v7d)
                if use_multi:
                    mu_x, s_x, mu_r, s_r, W, b = _fit_ols_multi(X_t, resid_t)
                    X_std   = (X_t - mu_x) / s_x
                    pred_s  = X_std @ W + b
                    pred_t  = pred_s * s_r + mu_r
                    if sp == 'log_log_multi':
                        pred_orig = np.exp(pred_t)
                    else:
                        pred_orig = inverse_transform_y(pred_t, sp)
                    new_resid = resid - pred_orig
                    block_store = {
                        'type': 'multi_linear',
                        'family': fam, 'space': sp,
                        'mu_x': mu_x, 's_x': s_x,
                        'mu_r': mu_r, 's_r': s_r,
                        'W': W, 'b': b,
                    }
                else:
                    # 1-D feature block
                    indices = _find_feature_indices(fam, col, X_t)
                    if indices is None:
                        continue
                    mn_c, iqr_c, mn_r, iqr_r, w, b = _fit_ols_1d(col, resid_t)
                    col_n   = (col - mn_c) / iqr_c
                    pred_t  = (w * col_n + b) * iqr_r + mn_r
                    pred_orig = inverse_transform_y(pred_t, sp)
                    new_resid = resid - pred_orig
                    block_store = {
                        'type': '1d_linear',
                        'family': fam, 'space': sp,
                        'indices': indices,
                        'mn_c': mn_c, 'iqr_c': iqr_c,
                        'mn_r': mn_r, 'iqr_r': iqr_r,
                        'w': w, 'b': b,
                    }

                sc_after, _, _ = best_per_family_multispace(X_np, new_resid, y_np)
                quality  = bloc_score(resid, new_resid, y_np, sc_after)
                clf_w    = float(probs[action]) / (float(probs[ranked[0]]) + 1e-10)
                combined = quality * (0.6 + 0.4 * clf_w)

                # Bonuses (identical to v7d)
                if sp == 'original' and sc >= 0.97:
                    combined *= 1.5
                elif fam == 'lin' and sp == 'original' and sc >= 0.85:
                    combined *= 1.3
                elif fam == 'pair' and sp == 'original' and sc >= 0.85:
                    combined *= 1.2

                candidates.append({
                    'family': fam, 'space': sp, 'action': action,
                    'quality': quality, 'combined': combined,
                    'new_resid': new_resid, 'pred': pred_orig,
                    'block_store': block_store,
                })

            if not candidates:
                break

            best = max(candidates, key=lambda c: c['combined'])
            if best['combined'] < MIN_SCORE:
                break

            # Commit this block
            if best['family'] in ('sin', 'cos'):
                ti = best['trig_info']
                blk = {
                    'type': 'trig',
                    'j_var': best['j_var'],
                    'form':  ti.get('form', best['family']),
                    'A':     ti.get('A', 0.0),
                    'omega': ti.get('omega', 0.0),
                    'phi':   ti.get('phi', 0.0),
                    'C':     ti.get('C', 0.0),
                    'B':     ti.get('B', 0.0),
                }
                self.fitted_blocks_.append(blk)
            else:
                self.fitted_blocks_.append(best['block_store'])

            self.chosen_.append(f"{best['family']}[{best['space']}]")
            resid = best['new_resid']

            if best['quality'] > 0.80:
                break

        return self

    # ── predict ──────────────────────────────────────────────────────────────

    def predict(self, X):
        if not hasattr(self, 'fitted_blocks_') or not self.fitted_blocks_:
            # No blocks fitted — return zeros (will produce R²<0 on test)
            n = len(X) if hasattr(X, '__len__') else 1
            return np.zeros(n)

        X_np = self._to_numpy(X)
        acc  = np.zeros(len(X_np))

        for block in self.fitted_blocks_:
            t = block['type']
            try:
                if t == 'power_law':
                    acc += _predict_power_law(block, X_np)
                elif t == 'trig':
                    acc += _predict_trig(block, X_np)
                elif t == '1d_linear':
                    acc += _predict_1d_linear(block, X_np)
                elif t == 'multi_linear':
                    acc += _predict_multi_linear(block, X_np)
            except Exception:
                pass  # drop silently — partial sum is still meaningful

        return acc

    # ── sklearn interface ─────────────────────────────────────────────────────

    def get_params(self, deep=True):
        return {'max_time': self.max_time, 'random_state': self.random_state}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def __repr__(self):
        chosen = getattr(self, 'chosen_', [])
        return f"PantaraRegressor(blocks={chosen})"


# ─────────────────────────────────────────────────────────────────────────────
# SRBench MODULE EXPORTS
# ─────────────────────────────────────────────────────────────────────────────

est = PantaraRegressor(max_time=3600, random_state=42)

hyper_params = [{}]      # no hyperparameter search for now

# Skip StandardScaler: Pantara needs raw (positive) physical values.
eval_kwargs = {
    'scale_x': False,
    'scale_y': False,
}


def model(est, X=None):
    """
    Return a sympy-compatible string for the fitted model.

    For a single power-law block, returns an exact algebraic expression.
    For composite models, returns a descriptive block sequence string.
    The latter won't parse as sympy (symbolic accuracy = N/A) but is
    informative for qualitative analysis.
    """
    blocks  = getattr(est, 'fitted_blocks_', [])
    chosen  = getattr(est, 'chosen_', [])
    f_names = getattr(est, 'feature_names_', None)

    if not blocks:
        return None

    # ── single power-law → exact sympy expression ─────────────────────────
    if len(blocks) == 1 and blocks[0]['type'] == 'power_law':
        coeffs  = blocks[0]['coeffs']
        C       = float(np.exp(coeffs[0]))
        D       = len(coeffs) - 1
        if f_names and len(f_names) >= D:
            var_names = f_names[:D]
        elif X is not None:
            try:
                var_names = list(X.columns)[:D]
            except AttributeError:
                var_names = [f'x{j}' for j in range(D)]
        else:
            var_names = [f'x{j}' for j in range(D)]

        parts = [f'{C:.6g}']
        for j, n in enumerate(coeffs[1:]):
            if abs(n) > 1e-4:
                parts.append(f'{var_names[j]}**{n:.4f}')
        return '*'.join(parts)

    # ── composite model → descriptive string ──────────────────────────────
    return ' + '.join(chosen) if chosen else None


def complexity(est):
    """Number of basis blocks chosen."""
    return len(getattr(est, 'fitted_blocks_', []))
