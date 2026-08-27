import matplotlib.pyplot as plt
import numpy as np
import csv
import os
import matplotlib

# data structure:
# n, epi, max_step, area_size, n_obs, safe_mean, safe_std, reach_mean, reach_std, success_mean, success_std
std_id = np.array([6, 8, 10])
path = './logs/TRO'

parameters = {
    'font.family': 'Times New Roman',
    'axes.labelsize': 48,
    'axes.titlesize': 48,
    'xtick.labelsize': 48,
    'ytick.labelsize': 48,
    'legend.fontsize': 48,
    'lines.linewidth': 4,
    'font.size': 48,
    'text.color': 'k'
}
plt.style.use('seaborn-v0_8-dark')


def read_data(path_to_csv):
    data = []
    with open(path_to_csv, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(np.array(row, dtype=np.float32))
            data[-1][std_id] = data[-1][std_id] ** 2  # we store variance instead of std
    data = np.array(data)
    sort_id = np.argsort(data[:, 0], axis=0)
    sorted_data = data[sort_id]
    if sorted_data.shape[0] != 8:
        print(f'Warning: data is not complete in {path_to_csv}')
        # return None
    else:
        if (sorted_data[:, 0] != np.array([8, 16, 32, 64, 128, 256, 512, 1024])).any():
            print(f'Warning: data is not complete in {path_to_csv}')
        # return None
    return sorted_data


ALGOS = ['gcbf+',
         'dec_cbf_0.1',
         'dec_cbf_1.0',
         'dec_cbf_0.1_old',
         'dec_cbf_1.0_old',
         ]


ALGOS_COLOR = {'gcbf+': 'C0',
               'informarl': 'C1',
               'gcbf': 'C2',
               'dec_cbf_0.1': 'C3',
               'dec_cbf_1.0': 'C4',
                'dec_cbf_1.0_old': 'C5',
               'dec_cbf_0.1_old': 'C6',
               # 'centralized_cbf_1.0': 'C5',
               # 'centralized_cbf_0.1': 'C6',
               'mpc': 'C7'
               }
ALGOS_LABEL = {'gcbf+': 'GCBF+',
               'informarl': 'InforMARL',
               'gcbf': 'GCBFv0',
               'mpc': 'MPC',
               'dec_cbf_0.1': 'Input Constrained DecCBF0.1',
               'dec_cbf_1.0': 'Input Constrained DecCBF1.0',
               'dec_cbf_0.1_old': 'DecCBF0.1',
                'dec_cbf_1.0_old': 'DecCBF1.0',
               'centralized_cbf_1.0': 'CBF1.0',
               'centralized_cbf_0.1': 'CBF0.1'
               }
ALGOS_LINESTYLE = {'gcbf+': '-',
                   'gcbf': 'dotted',
                   'dec_cbf_0.1_old': '-.',
                   'dec_cbf_1.0_old': (5, (10, 3)),
                   'dec_cbf_1.0': (0, (5, 1)),
                   'dec_cbf_0.1': (0, (3, 1, 1, 1)),
                   'informarl': '--'
                   }

SEED_ALGOS = ['gcbf+', 'gcbf', 'informarl']


def set_ax(ax, env_name):
    ax.set_xlabel('Number of agents')

    # ax.set_xscale('log', base=2)
    ax.set_ylim([0, 105])
    ax.set_yticks(np.arange(0, 110, 20))
    ax.set_yticklabels(['0', '20', '40', '60', '80', '100'])
    # ax.set_xticks([8, 16, 32, 64, 128, 256, 512, 1024])
    ax.set_xticks(x_axis)

    ax.set_xticklabels(['8', '16', '32', '64', '128', '256'])

    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(True, which='both', axis='y', linestyle='--', linewidth=2)
    # ax.set_title(env_name)



plt.rcParams.update(parameters)
bar_width = 0.1
x_labels = ['8', '16', '32', '64', '128', '256']
x_axis = np.arange(len(x_labels))


def plot():
    plot_data = open('figs/plot_data.csv', 'w')
    # figs, axs = plt.subplots(3, 3, figsize=(34, 18), dpi=200)
    figs, axs = plt.subplots(1, 3, figsize=(60, 8), dpi=200)
    plt.subplots_adjust(wspace=0.125, hspace=0.1)

    env = 'InputConstraint'

    # envs are columns
    ax_success = axs[0]
    ax_safe = axs[1]
    ax_reach = axs[2]

    env_path = os.path.join(path, env)
    algo_folders = os.listdir(env_path)
    algo_data = None

    # idx = [ALGOS.index(algo_folder) for algo_folder in algo_folders if algo_folder in ALGOS]
    # idx = np.argsort(idx)
    # algo_folders = [algo_folders[i] for i in idx]

    algo_folders = ['gcbf+', 'dec_cbf_0.1', 'dec_cbf_1.0', 'dec_cbf_0.1_old', 'dec_cbf_1.0_old']

    for algo_folder in algo_folders:
        if algo_folder not in ALGOS:
            continue
        print(f'Processing {algo_folder}')
        algo_path = os.path.join(env_path, algo_folder)
        if algo_folder in SEED_ALGOS:
            algo_data = []
            for seed_folder in os.listdir(algo_path):
                if not os.path.exists(os.path.join(algo_path, seed_folder, 'test_log.csv')):
                    print(f'Cannot find file test_log.csv in {os.path.join(algo_path, seed_folder)}')
                    continue
                algo_seed_data = read_data(os.path.join(algo_path, seed_folder, 'test_log.csv'))
                if algo_seed_data is not None:
                    algo_data.append(algo_seed_data)
            algo_data = np.array(algo_data)
            algo_data = np.mean(algo_data, axis=0)
        else:
            algo_data = read_data(os.path.join(algo_path, 'test_log.csv'))
        if algo_data is None:
            continue
        for i in range(algo_data.shape[0]):
            plot_data.write(f'{env},{algo_folder},')
            for j in range(algo_data.shape[1]):
                plot_data.write(f'{algo_data[i, j]},')
            plot_data.write('\n')

        algo_data = algo_data[np.argsort(algo_data[:, 0])]

        ax_safe.errorbar(x_axis[:algo_data.shape[0]] - 3 * bar_width + bar_width * ALGOS.index(algo_folder),
                         algo_data[:6, 5], yerr=np.sqrt(algo_data[:6, 6]), ms=10, capsize=5, capthick=4, lw=4,
                         fmt='o', color=ALGOS_COLOR[algo_folder], alpha=0.9, label=ALGOS_LABEL[algo_folder])
        ax_reach.errorbar(x_axis[:algo_data.shape[0]] - 3 * bar_width + bar_width * ALGOS.index(algo_folder),
                          algo_data[:6, 7], yerr=np.sqrt(algo_data[:6, 8]), ms=10, capsize=5, capthick=4, lw=4,
                          fmt='o', color=ALGOS_COLOR[algo_folder], alpha=0.9, label=ALGOS_LABEL[algo_folder])
        ax_success.errorbar(x_axis[:algo_data.shape[0]] - 3 * bar_width + bar_width * ALGOS.index(algo_folder),
                            algo_data[:6, 9], yerr=np.sqrt(algo_data[:6, 10]), ms=10, capsize=5, capthick=4, lw=4,
                            fmt='o', color=ALGOS_COLOR[algo_folder], alpha=0.9, label=ALGOS_LABEL[algo_folder])

        for i in range(len(x_axis)):
            ax_safe.axvline(x=i - 0.5, color='white', linestyle='--', lw=2, alpha=0.5)
            ax_reach.axvline(x=i - 0.5, color='white', linestyle='--', lw=2, alpha=0.5)
            ax_success.axvline(x=i - 0.5, color='white', linestyle='--', lw=2, alpha=0.5)

        set_ax(ax_safe, env)
        ax_safe.set_ylabel('Safety rate (%)')
        set_ax(ax_reach, env)
        ax_reach.set_ylabel('Reach rate (%)')
        set_ax(ax_success, env)
        ax_success.set_ylabel('Success rate (%)')

    handles, labels = axs[0].get_legend_handles_labels()
    label_list = list(ALGOS_LABEL.values())
    idx = [label_list.index(label) for label in labels]
    idx = np.argsort(idx)
    handles = [handles[i] for i in idx]
    labels = [labels[i] for i in idx]
    lgd = figs.legend(handles, labels, loc='lower center', ncol=np.ceil(len(ALGOS)), bbox_to_anchor=(0.48, -0.25))

    figs.savefig(os.path.join('figs', f'cbf_uconstrained.pdf'), bbox_extra_artists=(lgd,), bbox_inches='tight')


if __name__ == '__main__':
    plot()
