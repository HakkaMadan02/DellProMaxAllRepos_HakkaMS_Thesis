import numpy as np
import matplotlib.pyplot as plt


def plot_hw():
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

    data = np.load("min_dists.npy")

    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=300)
    ax.hist(data - 2 * 0.13, bins=10, color="C3")
    plt.axvline(x=0., color='k', lw=3)
    ax.set_xlabel("Min distance (m)")
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("logs/TRO/hw_min_dists.pdf")
    plt.show()


if __name__ == "__main__":
    plot_hw()
