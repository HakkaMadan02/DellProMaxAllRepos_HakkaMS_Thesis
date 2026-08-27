import ipdb
import jax
import matplotlib.pyplot as plt
import numpy as np

from model.env.cf_hl import CFHL


def main():
    n_agents = 3
    area_size = 4.0
    env = CFHL(n_agents, area_size)

    x0 = np.zeros(12)
    KK = 300

    x = x0
    T_x, T_ull, T_uhl = [x], [], []

    rk4_single = jax.jit(env.rk4_single)

    np.set_printoptions(linewidth=300)

    x_goal = np.zeros(12)
    x_goal[:3] = np.array([3.0, 2.0, 1.0])

    for kk in range(KK):
        # u_hl = np.zeros(4)
        #
        # if kk < 50:
        #     u_hl[CFHL.L_U_VX] = 1.0
        # elif kk < 100:
        #     u_hl[CFHL.L_U_VX] = 1.0
        #     u_hl[CFHL.L_U_VY] = 1.0
        # elif kk < 200:
        #     u_hl[CFHL.L_U_VX] = 1.0
        #     u_hl[CFHL.L_U_VZ] = 1.0
        # elif kk < 250:
        #     u_hl[CFHL.L_U_VX] = 1.0
        #     u_hl[CFHL.L_U_R] = 1.0

        # elif kk < 150:
        #     u_hl[CFHL.L_U_VX] = 1.0
        # elif kk < 200:
        #     u_hl[CFHL.L_U_VX] = 1.0
        # elif kk < 250:
        #     u_hl[CFHL.L_U_VY] = 1.0
        # elif kk < 300:
        #     u_hl[CFHL.L_U_VY] = 1.0

        u_hl = env.u_ref_inner_single(x, x_goal)
        x, u_ll = rk4_single(x, u_hl)

        # u_ll = env._get_ll_controls(x, u_hl)

        # xdot = env._agent_xdot_single_agent(x0, u_ll)
        # print(xdot)
        # print(xdot * env.dt)
        # print("x:")
        # print(x)
        # ipdb.set_trace()

        T_x.append(x)
        T_ull.append(u_ll)
        T_uhl.append(u_hl)

    T_x = np.stack(T_x, axis=0)
    T_ull = np.stack(T_ull, axis=0)
    T_uhl = np.stack(T_uhl, axis=0)

    nrows = 19
    figsize = np.array([10.0, 1.5 * nrows])
    fig, axes = plt.subplots(nrows, figsize=figsize, layout="constrained", sharex=True)

    for ii, ax in enumerate(axes[:3]):
        ax.plot(np.gradient(T_x[:, ii], env.dt), color="C0", marker="o", lw=0.5, ms=1.0)
        ax.set_ylabel(f"v{env.x_labels[ii]}", rotation=0, ha="right")

    for ii, ax in enumerate(axes[3:7]):
        ax.plot(T_uhl[:, ii], color="C0", marker="o", lw=0.5, ms=1.0)
        ax.set_ylabel(env.uhl_labels[ii], rotation=0, ha="right")

    for ii, ax in enumerate(axes[7:]):
        ax.plot(T_x[:, ii], color="C1", marker="o", lw=0.5, ms=1.0)
        ax.set_ylabel(env.x_labels[ii], rotation=0, ha="right")
    fig.savefig("test_cf_hl.pdf", bbox_inches="tight")


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
