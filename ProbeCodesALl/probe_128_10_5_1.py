from flax import linen as nn
import jax.numpy as jnp

class MLP128_10_5_1(nn.Module):
    features: int = 128

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(10)(x)
        x = nn.leaky_relu(x, negative_slope=0.01)
        x = nn.Dense(5)(x)
        x = nn.leaky_relu(x, negative_slope=0.01)
        x = nn.Dense(1)(x)
        x = nn.sigmoid(x)
        return x.squeeze(-1)