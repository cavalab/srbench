from typing import List, Tuple, Callable
import sympy as sy
import jax.numpy as jnp
from . import custom_functions


f_dict_jax = {
    "sin": (jnp.sin, 1),
    "cos": (jnp.cos, 1),
    "id": (custom_functions.identity, 1),
    "mul": (jnp.multiply, 2),
    "div": (custom_functions.div, 2),
    "sqrt": (custom_functions.sqrt_jax, 1),
    "exp": (jnp.exp, 1),
    "log": (custom_functions.log_jax, 1),
    "square": (custom_functions.square, 1),
    "cube": (custom_functions.cube, 1),
}


def dress_funs(functions):
    """
    Transfrom to jax function object and add arity
    """
    return [f_dict_jax[f] for f in functions]


def get_indices(
    functions: List[Tuple[Callable, int]]
) -> Tuple[List[int], List[List[int]]]:
    """
    Gets a list of function and then assigns to each function
    an index to act on. Returns indices for unary and binary
    separately.
    """
    unary_indices = []
    binary_indices = []
    i = 0

    # augment index by arity of operator
    for _, a in functions:
        if a == 1:
            unary_indices.append(i)
            i += 1
        elif a == 2:
            binary_indices.append([i, i + 1])
            i += 2
        else:
            raise KeyError
    return unary_indices, binary_indices


def get_una_bin_funs(functions):
    functions = dress_funs(functions)
    una_idx, bin_idx = get_indices(functions)

    unary_funcs = [
        (func, index)
        for func, index in zip((f for f, a in functions if a == 1), una_idx)
    ]
    binary_funcs = [
        (func, index)
        for func, index in zip((f for f, a in functions if a == 2), bin_idx)
    ]

    return unary_funcs, binary_funcs


def round_floats(expr, to: int = 3):
    """
    Takes a sympy expression and rounds every float to
    `to` digits
    """
    new_expr = expr
    # walks along the expression tree
    for a in sy.preorder_traversal(expr):
        if isinstance(a, sy.Float):
            new_expr = new_expr.subs(a, round(a, to))
    return new_expr
