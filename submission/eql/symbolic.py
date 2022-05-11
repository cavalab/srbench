import sympy as sy
import jax.numpy as jnp
import custom_functions
from utils import get_indices, get_una_bin_funs
from l0_dense import L0Dense
from typing import List

f_dict_sy = {
    jnp.sin: sy.sin,
    jnp.cos: sy.cos,
    custom_functions.identity: sy.Id,
    jnp.multiply: sy.Symbol.__mul__,
    custom_functions.div: sy.Symbol.__truediv__,
    jnp.exp: sy.exp,
    custom_functions.sqrt_jax: custom_functions.sqrt_sy,
    custom_functions.log_jax: custom_functions.log_sy,
    custom_functions.square: custom_functions.square,
    custom_functions.cube: custom_functions.cube,
}


def get_symbolic_expr_layer(W, b, functions, var_name="x"):
    """
    Constructs a sympy representation of the function described
    by the layer
    """
    in_features = W.shape[0]
    out_features = W.shape[1]

    unary_funcs, binary_funcs = get_una_bin_funs(functions)

    in_symbols = sy.symbols("{}:{}".format(var_name, in_features))
    z = []

    for i in range(out_features):
        o = 0
        for j in range(in_features):
            o += in_symbols[j] * W[j, i].item()
        if b is not None:
            o += b[i].item()
            z.append(o)

    outs = []
    for f, i in unary_funcs:
        s = f_dict_sy[f](z[i])
        outs.append(s)

    for f, i in binary_funcs:
        s = f_dict_sy[f](z[i[0]], z[i[1]])
        outs.append(s)
    return outs


def get_Wb(layer, use_l0=False):
    kernel = layer["kernel"]
    bias = layer["bias"]

    if use_l0:
        qz = layer["qz_loga"]
        # z = quantile_concrete(qz)
        z = L0Dense.deterministic_mask(qz)
        return z * kernel, bias

    return kernel, bias


def get_symbolic_expr(params, functions, use_l0=False):
    # *hidden, last = params["params"].values()
    last, *hidden = params["params"].values()
    # hidden = [h["linear_layer"] for h in hidden]

    if not isinstance(functions[0], List):
        functions = [functions] * len(hidden)

    a = get_symbolic_expr_layer(
        *get_Wb(hidden[0]["linear_layer"], use_l0), functions[0]
    )

    # skip first hidden layer
    for i in range(1, len(hidden)):
        b = get_symbolic_expr_layer(
            *get_Wb(hidden[i]["linear_layer"], use_l0), functions[i], var_name="b"
        )
        c = []
        for k in range(len(b)):
            c.append(b[k].subs({"b" + str(j): a[j] for j in range(len(a))}))
        a = c

    # get weight/bias of last (normal) linear layer
    w, b = last["kernel"], last["bias"]
    z = []

    in_features = w.shape[0]
    out_features = w.shape[1]

    for i in range(out_features):
        s = 0
        for j in range(in_features):
            s += a[j] * w[j, i].item()
        if b is not None:
            s += b[i].item()
        z.append(s)
    return z
