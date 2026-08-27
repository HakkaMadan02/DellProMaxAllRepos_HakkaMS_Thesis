from jax.experimental import io_callback
from functools import partial
import jax.numpy as jnp
import numpy as np
import jax.lax as lax

def append_to_csv(x):
    with open("testfile.csv", "a") as f:
        f.write("{}\n".format(x))

def body(carry, inp):
    io_callback(append_to_csv, None, carry)
    return carry + 1, carry

xx = np.arange(100)
c0 = np.array(1.0)
_, _ = lax.scan(body, c0, xx)
