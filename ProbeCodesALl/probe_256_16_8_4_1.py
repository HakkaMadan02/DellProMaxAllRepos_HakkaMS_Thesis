import jax.numpy as jnp
from flax import linen as nn

class MLP128_16_8_4_1(nn.Module):
    features: int = 128

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(16)(x)
        x = nn.leaky_relu(x, negative_slope=0.01)
        x = nn.Dense(8)(x)
        x = nn.leaky_relu(x, negative_slope=0.01)
        x = nn.Dense(4)(x)
        x = nn.leaky_relu(x, negative_slope=0.01)
        x = nn.Dense(1)(x)
        x = nn.sigmoid(x)
        return x.squeeze(-1)
