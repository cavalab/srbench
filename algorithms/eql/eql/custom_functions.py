import jax.numpy as jnp
import sympy as sy
import jax


def identity(x):
    return x


def div(x, y):
    return x / (jnp.abs(y) + 1e-5)


def sqrt_jax(x):
    return jnp.sqrt(jnp.abs(x) + 1e-8)


def sqrt_sy(x):
    return sy.sqrt(sy.Abs(x))


def log_jax(x):
    return jnp.log(jnp.abs(x) + 1e-8)


def log_sy(x):
    return sy.log(abs(x))


def square(x):
    return x * x


def cube(x):
    return x * x * x
