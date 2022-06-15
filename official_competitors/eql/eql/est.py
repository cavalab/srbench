from sklearn.base import BaseEstimator, RegressorMixin
import sklearn
from . import eqlearner as eql
from .symbolic import get_symbolic_expr, get_symbolic_expr_layer
import jax
from jax import random, numpy as jnp
from .np_utils import flatten, unflatten
import numpy as np
import scipy
import time
from sklearn.metrics import r2_score
from .sy_utils import simplicity, round_floats
import optax



class EQL(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        n_layers=1,
        functions="id;mul",
        n_iter=100,
        drop_rate=0.5,
        reg=1e-3,
        random_state=None,
        do_bfgs=True,
    ):

        self.n_layers = n_layers
        self.functions = functions
        self.n_iter = n_iter
        self.drop_rate = drop_rate
        self.reg = reg
        self.random_state = random_state
        self.do_bfgs = do_bfgs

    def fit(self, X, y):
        # comply with scikit
        if np.iscomplex(X).any():
            raise ValueError("Complex data not supported")
        self.n_features_in_ = X.shape[-1]

        # sklearn.utils.check_array(X)
        # sklearn.utils.check_array(y)

        # we only deal with floats
        X = X.astype(float)
        y = y.astype(float)

        # first get number of features
        if y.ndim == 1:
            out_feat = 1
        else:
            out_feat = y.shape[-1]

        self._fn_list = self.functions.split(";")
        self._eql = eql.EQL(
            n_layers=self.n_layers,
            functions=self._fn_list,
            features=out_feat,
            use_l0=True,
            drop_rate=self.drop_rate,
        )

        def make_l0_fn():
            def l0(params, key):
                return self._eql.apply(
                    params, rngs={"l0": key}, method=self._eql.l0_reg
                )

            return jax.jit(l0)

        def mse_fn(params, key):
            def err(x, y):
                pred = self._eql.apply(params, x, rngs={"l0": key})
                return (pred - y) ** 2

            return jnp.mean(jax.vmap(err)(X, y))

        mse_fn = jax.jit(mse_fn)
        l0_fn = make_l0_fn()

        def loss(params, key):
            return mse_fn(params, key) + self.reg * l0_fn(params, key)

        loss_grad_fn = jax.jit(jax.value_and_grad(loss))

        if self.random_state == None:
            self.random_state = np.random.randint(0, 9999)

        key = random.PRNGKey(self.random_state)
        key, k1, k2 = random.split(key, 3)
        params = self._eql.init({"params": k1, "l0": k2}, X)

        tx = optax.adam(learning_rate=1e-2)
        opt_state = tx.init(params)

        for i in range(self.n_iter):
            key, _ = random.split(key)
            loss_val, grads = loss_grad_fn(params, key)
            updates, opt_state = tx.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            
        if self.do_bfgs:    
            def np_fn(params, key):
                params = unflatten(spec, params)
                # new key or same key
                #key = jax.random.fold_in(key, np_fn.counter)
                loss, grad = loss_grad_fn(params, key)
                _, grad = flatten(grad)
                np_fn.counter += 1
                return loss, np.array(grad)
            np_fn.counter = 0
            
            spec, flat_params = flatten(params)
            
            # final fitting
            x0, _, info = scipy.optimize.fmin_l_bfgs_b(
                    np_fn,
                    args=[key],
                    x0=np.array(flat_params),
                    maxfun=50,
                    factr=1,
                    m=20,
                    pgtol=1e-14,
            )

            self._params = unflatten(spec, x0)
            
        else: 
            self._params = params
        
        return self

    def get_score(self, X, y, param):
        yhat = jnp.nan_to_num(self._eql.apply(param, X, deterministic=True))
        exprs = self.get_eqn()
        # exprs might be invalid
        try:
            simps = np.array([simplicity(e) for e in exprs])
        except TypeError:
            simps = -999.0

        r2 = r2_score(yhat, y)
        return r2 + np.clip(np.mean(simps), a_min=0.0, a_max=10.0)

    def predict(self, X):
        yhat = self._eql.apply(self._params, X, deterministic=True)
        return yhat

    def score(self, X, y):
        return self.get_score(X, y, self._params)

    def get_eqn(self):
        # symb is list of len = dim(y_out)
        # for now assume 1d output
        symb = get_symbolic_expr(self._params, self._fn_list, use_l0=True)
        return round_floats(symb[0])
        #return [round_floats(s) for s in symb]


