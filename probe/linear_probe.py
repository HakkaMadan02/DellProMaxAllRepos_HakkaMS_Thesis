import jax
import jax.numpy as jnp
from flax import linen as nn
import optax

class LinearProbe(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(1, use_bias = True)(x)
        x = nn.sigmoid(x)
        return x.squeeze(-1)