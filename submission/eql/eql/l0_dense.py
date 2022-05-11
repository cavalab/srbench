import jax
from typing import Any, Callable, Sequence, Optional
from jax import lax, random, numpy as jnp
import flax
from flax import linen as nn


limit_a, limit_b, epsilon = -0.1, 1.1, 1e-6


def init_qz_loga(drop_rate, stddev=1e-2, dtype: Any = jnp.float_) -> Callable:
    mean = jnp.log(1 - drop_rate) - jnp.log(drop_rate)

    def init(key, shape, dtype=jnp.float_):
        return random.normal(key, shape, dtype) * stddev + mean

    return init


def hard_tanh(x):
    return jnp.where(x > 1, 1, jnp.where(x < 0, 0, x))


# def quantile_concrete(x, qz_loga, temperature=2.0 / 3.0):
#     y = nn.sigmoid((jnp.log(x) - jnp.log(1 - x) + qz_loga) / temperature)
#     return y * (limit_b - limit_a) + limit_a


class L0Dense(nn.Module):
    features: int
    drop_rate: float = 0.5
    temperature: float = 2.0 / 3.0
    kernel_init: Callable = nn.initializers.lecun_normal()
    bias_init: Callable = nn.initializers.zeros

    def quantile_concrete(self, x, qz_loga):
        y = nn.sigmoid((jnp.log(x) - jnp.log(1 - x) + qz_loga) / self.temperature)
        return y * (limit_b - limit_a) + limit_a

    def cdf_qz(self, x):
        qz_loga = self.variables["params"]["qz_loga"]
        xn = (x - limit_a) / (limit_b - limit_a)
        logits = jnp.log(xn) - jnp.log(1 - xn)
        return lax.clamp(
            epsilon, nn.sigmoid(logits * self.temperature - qz_loga), 1.0 - epsilon
        )

    def l0_reg(self):
        return jnp.sum((1 - self.cdf_qz(0)))

    @staticmethod
    def deterministic_mask(qz_loga):
        pi = nn.sigmoid(qz_loga)
        mask = hard_tanh(pi * (limit_b - limit_a) + limit_a)
        return mask

    def sample_mask(self, qz_loga, rng):
        shape = qz_loga.shape
        eps = random.uniform(rng, shape, minval=epsilon, maxval=1.0 - epsilon)
        z = self.quantile_concrete(eps, qz_loga)
        mask = hard_tanh(z)
        return mask

    @nn.compact
    def __call__(self, inputs, deterministic: Optional[bool] = False):
        kernel = self.param(
            "kernel", self.kernel_init, (inputs.shape[-1], self.features)
        )

        qz_loga = self.param(
            "qz_loga",
            init_qz_loga(self.drop_rate),
            # (inputs.shape[-1],))
            kernel.shape,
        )

        if deterministic:
            pi = nn.sigmoid(qz_loga)
            mask = hard_tanh(pi * (limit_b - limit_a) + limit_a)
            # mask = jnp.broadcast_to(z[:,None], kernel.shape)
            kernel = kernel * mask
        else:
            rng = self.make_rng("l0")
            # rng, _ = random.split(rng, 2)
            mask = self.sample_mask(qz_loga, rng)
            # mask = jnp.broadcast_to(mask[:,None], kernel.shape)
            kernel = kernel * mask

        y = lax.dot_general(
            inputs,
            kernel,
            (((inputs.ndim - 1,), (0,)), ((), ())),
        )

        bias = self.param("bias", self.bias_init, (self.features,))

        y = y + bias
        return y
