import numpy as np
import jax.numpy as jnp
import jax.tree_util as jtu
import functools as ft
import jax

from scipy.linalg import inv, solve_discrete_are
from typing import Callable

from ..utils.typing import Array, List, Pos2d, Radius, Pos3d, BoolScalar, Pos, State, Action
from ..utils.utils import merge01
from .obstacle import Obstacle, Rectangle, Cuboid, Sphere


def RK4_step(x_dot_fn: Callable, x: State, u: Action, dt: float) -> Array:
    k1 = x_dot_fn(x, u)
    k2 = x_dot_fn(x + 0.5 * dt * k1, u)
    k3 = x_dot_fn(x + 0.5 * dt * k2, u)
    k4 = x_dot_fn(x + dt * k3, u)
    return x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def lqr(
        A: np.ndarray,
        B: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
):
    """
    Solve the discrete time lqr controller.
        x_{t+1} = A x_t + B u_t
        cost = sum x.T*Q*x + u.T*R*u
    Code adapted from Mark Wilfred Mueller's continuous LQR code at
    https://www.mwm.im/lqr-controllers-with-python/
    Based on Bertsekas, p.151
    Yields the control law u = -K x
    """

    # first, try to solve the Riccati equation
    X = solve_discrete_are(A, B, Q, R)

    # compute the LQR gain
    K = inv(B.T @ X @ B + R) @ (B.T @ X @ A)

    return K


def get_lidar(start_point: Pos, obstacles: Obstacle, num_beams: int, sense_range: float, max_returns: int = 32):
    if isinstance(obstacles, Rectangle):
        thetas = jnp.linspace(-np.pi, np.pi - 2 * np.pi / num_beams, num_beams)
        starts = start_point[None, :].repeat(num_beams, axis=0)
        ends = jnp.stack(
            [starts[..., 0] + jnp.cos(thetas) * sense_range, starts[..., 1] + jnp.sin(thetas) * sense_range],
            axis=-1)
    elif isinstance(obstacles, Cuboid) or isinstance(obstacles, Sphere):
        thetas = jnp.linspace(-np.pi / 2 + 2 * np.pi / num_beams, np.pi / 2 - 2 * np.pi / num_beams, num_beams // 2)
        phis = jnp.linspace(-np.pi, np.pi - 2 * np.pi / num_beams, num_beams)
        starts = start_point[None, :].repeat(thetas.shape[0] * phis.shape[0] + 2, axis=0)

        def get_end_point(theta, phi):
            return jnp.array([
                start_point[0] + jnp.cos(theta) * jnp.cos(phi) * sense_range,
                start_point[1] + jnp.cos(theta) * jnp.sin(phi) * sense_range,
                start_point[2] + jnp.sin(theta) * sense_range
            ])

        def get_end_point_theta(theta):
            return jax.vmap(lambda phi: get_end_point(theta, phi))(phis)

        ends = merge01(jax.vmap(get_end_point_theta)(thetas))
        ends = jnp.concatenate([ends,
                                start_point[None, :] + jnp.array([[0., 0., sense_range]]),
                                start_point[None, :] + jnp.array([[0., 0., -sense_range]])], axis=0)
    else:
        raise NotImplementedError
    sensor_data = raytracing(starts, ends, obstacles, max_returns)

    return sensor_data

#
# def get_lidar3D(start_point: Pos3d, obstacles: Obstacle, num_beams: int, sense_range: float):
#     thetas = jnp.linspace(-np.pi, np.pi - 2 * np.pi / num_beams, num_beams)
#     phis = jnp.linspace(-np.pi / 2, np.pi / 2, num_beams)
#     starts = start_point[None, :].repeat(num_beams, axis=0)
#     ends = jnp.stack(
#         [starts[..., 0] + jnp.cos(thetas) * sense_range * jnp.cos(phis),
#             starts[..., 1] + jnp.sin(thetas) * sense_range * jnp.cos(phis),
#             starts[..., 2] + jnp.sin(phis) * sense_range],
#         axis=-1)
#
#     sensor_data = raytracing3D(starts, ends, obstacles)
#     return sensor_data


def inside_obstacles(points: Pos, obstacles: Obstacle, r: Radius = 0.) -> BoolScalar:
    """
    points: (n, n_dim) or (n_dim, )
    obstacles: tree_stacked obstacles.

    Returns: (n, ) or (,). True if in collision, false otherwise.
    """
    # one point inside one obstacle
    def inside(point: Pos, obstacle: Obstacle):
        return obstacle.inside(point, r)

    # one point inside any obstacle
    def inside_any(point: Pos, obstacle: Obstacle):
        return jax.vmap(ft.partial(inside, point))(obstacle).max()

    # any point inside any obstacle
    if points.ndim == 1:
        if obstacles.center.shape[0] == 0:
            return jnp.zeros((), dtype=bool)
        is_in = inside_any(points, obstacles)
    else:
        if obstacles.center.shape[0] == 0:
            return jnp.zeros(points.shape[0], dtype=bool)
        is_in = jax.vmap(ft.partial(inside_any, obstacle=obstacles))(points)

    return is_in


def raytracing(starts: Pos, ends: Pos, obstacles: Obstacle, max_returns: int) -> Pos:
    # if the start point if inside the obstacle, return the start point
    is_in = inside_obstacles(starts, obstacles)

    def raytracing_single(start: Pos, end: Pos, obstacle: Obstacle):
        return obstacle.raytracing(start, end)

    def raytracing_any(start: Pos, end: Pos, obstacle: Obstacle):
        return jax.vmap(ft.partial(raytracing_single, start, end))(obstacle).min()

    if obstacles.center.shape[0] == 0:
        alphas = jnp.ones(starts.shape[0]) * 1e6
    else:
        alphas = jax.vmap(ft.partial(raytracing_any, obstacle=obstacles))(starts, ends)
        alphas *= (1 - is_in)

    # assert max_returns <= alphas.shape[0]
    alphas_return = jnp.argsort(alphas)[:max_returns]

    hitting_points = starts + (ends - starts) * (alphas[..., None])

    return hitting_points[alphas_return]
