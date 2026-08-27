import jax

import jax.numpy as jnp
import jax.random as jr


def test_gpu():
    a = jr.uniform(jr.PRNGKey(0), (25000, 25000))
    b = jr.uniform(jr.PRNGKey(1), (25000, 25000))

    def f(x, y):
        return jnp.matmul(x, y)

    def scan_f(carry, inp):
        aa, bb = inp
        return f(carry, aa), 0

    # f_vmap = jax.vmap(f)
    # c = f_vmap(a, b)

    #
    # c = jax.lax.scan(scan_f, jnp.zeros((1000, 1000)), (a, b))
    c = a
    for i in range(100000):
        c = f(c, b)

    return c


if __name__ == '__main__':
    test_gpu()
    print('done')
