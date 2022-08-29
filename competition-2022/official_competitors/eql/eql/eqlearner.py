import jax
from jax import lax, random, numpy as jnp
import sympy as sy
from flax import linen as nn
from typing import List, Tuple, Callable
from . import custom_functions
from .utils import get_indices, f_dict_jax
from .l0_dense import L0Dense
from typing import Optional


class EQLLayer(nn.Module):
    functions: List[str]

    # params for l0 layers
    use_l0: bool = False
    drop_rate: Optional[float] = 0.5
    temperature: float = 2.0 / 3.0

    def setup(self, **kwargs):
        # convert function string to function through dict
        self.jax_functions = [f_dict_jax[f] for f in self.functions]

        # sum arity of all functions for num of params
        self.number_of_vars = sum(a for f, a in self.jax_functions)
        self.number_of_functions = len(self.functions)

        # indices of cols that respective functions act on.
        # split unary and binary
        self.unary_indices, self.binary_indices = get_indices(self.jax_functions)

        # combine function and respective index into tuple
        self.unary_funcs = [
            (func, index)
            for func, index in zip(
                (f for f, a in self.jax_functions if a == 1), self.unary_indices
            )
        ]
        self.binary_funcs = [
            (func, index)
            for func, index in zip(
                (f for f, a in self.jax_functions if a == 2), self.binary_indices
            )
        ]

        self.num_unary_funcs = len(self.unary_funcs)
        self.num_binary_funcs = len(self.binary_funcs)

        self.features = self.num_unary_funcs + 2 * self.num_binary_funcs

        if self.use_l0:
            self.linear_layer = L0Dense(self.features, self.drop_rate, self.temperature)
        else:
            self.linear_layer = nn.Dense(self.features)

    def l0_reg(self):
        if self.use_l0:
            return self.linear_layer.l0_reg()
        else:
            raise Exception("Not using L0 reg")

    def __call__(self, inputs, deterministic=False):
        if self.use_l0:
            z = self.linear_layer(inputs, deterministic)
        else:
            z = self.linear_layer(inputs)

        unary_stack = jnp.stack([f(z[..., i]) for f, i in self.unary_funcs], -1)
        if self.binary_funcs:
            binary_stack = jnp.stack(
                [f(z[..., i[0]], z[..., i[1]]) for f, i in self.binary_funcs], -1
            )
            y = jnp.concatenate((unary_stack, binary_stack), -1)
        else:
            y = unary_stack
        return y


class EQL(nn.Module):
    n_layers: int
    functions: List
    features: int

    # params for l0 reg
    use_l0: Optional[bool] = None
    drop_rate: Optional[float] = 0.5
    temperature: float = 2.0 / 3.0

    def setup(self):
        # use the same function for all layers if there's just one list as input
        if not isinstance(self.functions[0], List):
            flist = [self.functions] * self.n_layers
        else:
            flist = self.functions

        self.layers = [
            EQLLayer(flist[i], self.use_l0, self.drop_rate, self.temperature)
            for i in range(self.n_layers)
        ]

        if self.use_l0:
            self.last = L0Dense(self.features, self.drop_rate, self.temperature)
        else:
            self.last = nn.Dense(self.features)

    def l0_reg(self):
        sum = 0.0
        for lyr in self.layers:
            sum += lyr.l0_reg()
        return sum + self.last.l0_reg()

    def __call__(self, inputs, deterministic: Optional[bool] = None):
        x = inputs
        if self.use_l0:
            for lyr in self.layers:
                x = lyr(x, deterministic)
            return self.last(x, deterministic)
        else:
            for lyr in self.layers:
                x = lyr(x)
            return self.last(x)
