import functools as ft
import pathlib
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import scipy

from typing import NamedTuple, Tuple, Optional
from jax.lax import while_loop
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d import Axes3D, proj3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from ..trainer.utils import centered_norm, np2jax
from ..utils.graph import EdgeBlock, GetGraph, GraphsTuple
from ..utils.typing import ABool, Action, AgentState, Array, Cost, Done, Info, Pos3d, Reward, State
from ..utils.utils import MutablePatchCollection, merge01, save_anim, tree_index
from .base import MultiAgentEnv, RolloutResult
from .obstacle import Obstacle, Cuboid
from .plot import plot_graph, get_BuRd, get_cuboid_collection
from .utils import get_lidar, inside_obstacles, lqr
from .linear_drone import LinearDrone


class MAV(LinearDrone):
    AGENT = 0
    GOAL = 1
    OBS = 2

    class EnvState(NamedTuple):
        agent: AgentState
        goal: State
        obstacle: Obstacle

        @property
        def n_agent(self) -> int:
            return self.agent.shape[0]

    EnvGraphsTuple = GraphsTuple[State, EnvState]

    PARAMS = {
        "area_size": 10.0,
        "max_step": 256,
        "drone_radius": 0.25,
        "comm_radius": 0.5,
        "n_rays": 32,
        "obs_len_range": [0.1, 0.4],
        "n_obs": 4
    }

    def __init__(self, num_agents: int, dt: float = 0.03, params: dict = None):
        super(MAV, self).__init__(num_agents, dt, params)
        self._A = np.zeros((self.state_dim, self.state_dim))
        self._A[0, 3] = 1.
        self._A[1, 4] = 1.
        self._A[2, 5] = 1.
        self._A[3, 3] = -5.371
        self._A[4, 4] = -3.573
        self._A[5, 5] = -9.009
        A_discrete = scipy.linalg.expm(self._A * self._dt)

        self._B = np.zeros((self.state_dim, self.action_dim))
        self._B[3, 0] = 1000.
        self._B[4, 1] = 1000.
        self._B[5, 2] = 1000.  # 6.0

        self._Q = np.diag([5e10, 5e10, 5e10, 1., 1., 1.])
        self._R = np.eye(self.action_dim) * 0.0001
        self._K = lqr(A_discrete, self._B, self._Q, self._R)
        self.create_obstacles = jax.vmap(Cuboid.create)
        self.n_rays = self.params['n_rays']
        print(self._K)


