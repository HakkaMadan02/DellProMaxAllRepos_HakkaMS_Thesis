import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import jax

from colour import hsl2hex
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.pyplot import Axes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import List, Optional, Tuple, Union

from ..utils.typing import EdgeIndex, Pos2d, Pos3d, Array
from ..utils.utils import merge01
from .obstacle import Cuboid, Sphere


def plot_graph(
        ax: Axes,
        pos: Pos2d,
        radius: Union[float, List[float]],
        color: Union[str, List[str]],
        with_label: Union[bool, List[bool]] = True,
        plot_edge: bool = False,
        edge_index: Optional[EdgeIndex] = None,
        edge_color: Union[str, List[str]] = 'k',
        alpha: float = 1.0,
        obstacle_color: str = '#000000',
) -> Axes:
    if isinstance(radius, float):
        radius = np.ones(pos.shape[0]) * radius
    if isinstance(radius, list):
        radius = np.array(radius)
    if isinstance(color, str):
        color = [color for _ in range(pos.shape[0])]
    if isinstance(with_label, bool):
        with_label = [with_label for _ in range(pos.shape[0])]
    circles = []
    for i in range(pos.shape[0]):
        circles.append(plt.Circle((float(pos[i, 0]), float(pos[i, 1])),
                                radius=radius[i], color=color[i], clip_on=False, alpha=alpha, linewidth=0.0))
        # ax.add_patch(plt.Circle((float(pos[i, 0]), float(pos[i, 1])),
        #                         radius=radius[i], color=color[i], clip_on=False, alpha=alpha, linewidth=0.0))
        if with_label[i]:
            ax.text(float(pos[i, 0]), float(pos[i, 1]), f'{i}', size=12, color="k",
                    family="sans-serif", weight="normal", horizontalalignment="center", verticalalignment="center",
                    transform=ax.transData, clip_on=True)
    circles = PatchCollection(circles, match_original=True)
    ax.add_collection(circles)

    if plot_edge and edge_index is not None:
        if isinstance(edge_color, str):
            edge_color = [edge_color for _ in range(edge_index.shape[1])]
        start, end = pos[edge_index[0, :]], pos[edge_index[1, :]]
        direction = (end - start) / jnp.linalg.norm(end - start, axis=1, keepdims=True)
        start = start + direction * radius[edge_index[0, :]][:, None]
        end = end - direction * radius[edge_index[1, :]][:, None]
        widths = (radius[edge_index[0, :]] + radius[edge_index[1, :]]) * 20
        lines = np.stack([start, end], axis=1)
        edges = LineCollection(lines, colors=edge_color, linewidths=widths, alpha=0.5)
        ax.add_collection(edges)
    return ax


def plot_node_3d(ax, pos: Pos3d, r: float, color: str, alpha: float, grid: int = 10) -> Axes:
    u = np.linspace(0, 2 * np.pi, grid)
    v = np.linspace(0, np.pi, grid)
    x = r * np.outer(np.cos(u), np.sin(v)) + pos[0]
    y = r * np.outer(np.sin(u), np.sin(v)) + pos[1]
    z = r * np.outer(np.ones(np.size(u)), np.cos(v)) + pos[2]
    ax.plot_surface(x, y, z, color=color, alpha=alpha)
    return ax


def plot_graph_3d(
        ax,
        pos: Pos3d,
        radius: float,
        color: Union[str, List[str]],
        with_label: bool = True,
        plot_edge: bool = False,
        edge_index: Optional[EdgeIndex] = None,
        edge_color: Union[str, List[str]] = 'k',
        alpha: float = 1.0,
        obstacle_color: str = '#000000',
):
    if isinstance(color, str):
        color = [color for _ in range(pos.shape[0])]
    for i in range(pos.shape[0]):
        plot_node_3d(ax, pos[i], radius, color[i], alpha)
        if with_label:
            ax.text(pos[i, 0], pos[i, 1], pos[i, 2], f'{i}', size=12, color="k", family="sans-serif", weight="normal",
                    horizontalalignment="center", verticalalignment="center")
    if plot_edge:
        if isinstance(edge_color, str):
            edge_color = [edge_color for _ in range(edge_index.shape[1])]
        for i_edge in range(edge_index.shape[1]):
            # for j in edge_index[1]:
            i = edge_index[0, i_edge]
            j = edge_index[1, i_edge]
            vec = pos[i, :] - pos[j, :]
            x = [pos[i, 0] - 2 * radius * vec[0], pos[j, 0] + 2 * radius * vec[0]]
            y = [pos[i, 1] - 2 * radius * vec[1], pos[j, 1] + 2 * radius * vec[1]]
            z = [pos[i, 2] - 2 * radius * vec[2], pos[j, 2] + 2 * radius * vec[2]]
            ax.plot(x, y, z, linewidth=1.0, color=edge_color[i_edge])
    return ax


def get_BuRd():
    # blue = "#3182bd"
    # blue = hsl2hex([0.57, 0.59, 0.47])
    blue = hsl2hex([0.57, 0.5, 0.55])
    light_blue = hsl2hex([0.5, 1.0, 0.995])

    # Tint it to orange a bit.
    # red = "#de2d26"
    # red = hsl2hex([0.04, 0.74, 0.51])
    red = hsl2hex([0.028, 0.62, 0.59])
    light_red = hsl2hex([0.098, 1.0, 0.995])

    sdf_cm = LinearSegmentedColormap.from_list("SDF", [(0, light_blue), (0.5, blue), (0.5, red), (1, light_red)], N=256)
    return sdf_cm


def get_faces_cuboid(points: Pos3d) -> Array:
    point_id = jnp.array([[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [0, 3, 7, 4], [1, 2, 6, 5]])
    faces = points[point_id]
    return faces


def get_cuboid_collection(
        obstacles: Cuboid, alpha: float = 0.8, linewidth: float = 1.0, edgecolor: str = 'k', facecolor: str = 'r'
) -> Poly3DCollection:
    get_faces_vmap = jax.vmap(get_faces_cuboid)
    obs_col = Poly3DCollection(
        merge01(get_faces_vmap(obstacles.points)),
        alpha=alpha,
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor
    )
    return obs_col


def get_sphere_collection(
        obstacles: Sphere, alpha: float = 0.8, facecolor: str = 'r'
) -> Poly3DCollection:

    def get_sphere(inp):
        center = inp[:3]
        radius = inp[3]
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi - 0.03, 30)
        x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
        y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
        z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
        return jnp.stack([x, y, z], axis=-1)

    get_sphere_vmap = jax.vmap(get_sphere)
    obs_col = Poly3DCollection(
        merge01(get_sphere_vmap(jnp.concatenate([obstacles.center, obstacles.radius[:, None]], axis=-1))),
        alpha=alpha,
        linewidth=0.0,
        edgecolor='k',
        facecolor=facecolor
    )

    return obs_col
