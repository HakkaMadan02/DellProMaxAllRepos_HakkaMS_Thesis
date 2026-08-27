import numpy as np
import matplotlib.pyplot as plt


def plot_reach_time():
    plt.style.use('seaborn-v0_8-dark')
    parameters = {
        'font.family': 'Times New Roman',
        'axes.labelsize': 36,
        'axes.titlesize': 36,
        'xtick.labelsize': 36,
        'ytick.labelsize': 36,
        'legend.fontsize': 36,
        'lines.linewidth': 4,
        'font.size': 36,
        'text.color': 'k',
        'mathtext.fontset': 'stix',
    }
    plt.rcParams.update(parameters)

    data = np.load("logs/TRO/DoubleIntegrator/gcbf+/seed0_20231123001301/reach_time.npy")
    data = data.flatten()
    data = data.astype(np.float32)

    data_nom = np.load("logs/TRO/DoubleIntegrator/reach_time_nom.npy")
    data_nom = data_nom.flatten()
    data_nom = data_nom.astype(np.float32)

    data = data / data_nom

    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=300)
    ax.hist(data, bins=200, color="C3", cumulative=True, density=True)
    plt.axvline(x=0., color='k', lw=3)
    ax.set_xlabel("Relative delay ratio")
    # ax.set_xscale("log")
    ax.set_ylabel("Proportion")
    # ax.set_ylim(0, 1)
    ax.set_xlim(1, 20)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.grid(True)
    plt.tight_layout()
    plt.savefig("logs/TRO/reach_time_doubint.pdf")
    plt.show()


if __name__ == "__main__":
    plot_reach_time()
