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
    'axes.labelsize': 84,
    'axes.titlesize': 84,
    'xtick.labelsize': 84,
    'ytick.labelsize': 84,
    'legend.fontsize': 72,
    'lines.linewidth': 8,
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


# ENVS = ['SingleIntegrator', 'DoubleIntegrator', 'LinearDrone', 'DubinsCar', 'CrazyFlie']
# ENVS = ['SingleIntegrator', 'DubinsCar', 'LinearDrone']
ENVS = ['CFHL']
ALGOS = ['gcbf+',
         'gcbf',
         'centralized_cbf_1.0',
         'centralized_cbf_0.1',
         'dec_cbf_1.0',
         'dec_cbf_0.1',
         'informarl'
         ]
ALGOS_COLOR = {'gcbf+': 'C0',
               'gcbf': 'C1',
               'centralized_cbf_1.0': 'C2',
               'centralized_cbf_0.1': 'C3',
               'dec_cbf_1.0': 'C4',
               'dec_cbf_0.1': 'C5',
               'informarl': 'C6'}
ALGOS_LABEL = {'gcbf+': 'GCBF+',
               'gcbf': 'GCBF',
               'centralized_cbf_1.0': 'CBF1.0',
               'centralized_cbf_0.1': 'CBF0.1',
               'dec_cbf_1.0': 'DecCBF1.0',
               'dec_cbf_0.1': 'DecCBF0.1',
               'informarl': 'InforMARL'}
SEED_ALGOS = ['gcbf+', 'gcbf', 'informarl']
ALGOS_LINESTYLE = {'gcbf+': '-',
                   'gcbf': 'dotted',
                   'centralized_cbf_1.0': '-.',
                   'centralized_cbf_0.1': (5, (10, 3)),
                   'dec_cbf_1.0': (0, (5, 1)),
                   'dec_cbf_0.1': (0, (3, 1, 1, 1)),
                   'informarl': '--'
                   }

plt.rcParams.update(parameters)


def set_ax(ax, env_name):
    ax.set_xlabel('Number of agents')
    ax.set_xscale('log', base=2)
    ax.set_ylim([0, 105])
    ax.set_yticks(np.arange(0, 110, 20))
    ax.set_yticklabels(['0', '20', '40', '60', '80', '100'])
    ax.set_xticks([8, 16, 32, 64, 128, 256, 512, 1024])
    ax.set_xticklabels(['8', '16', '32', '64', '128', '256', '512', '1024'])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(True, which='both', axis='both', linestyle='--')
    ax.set_title(env_name)


# one figure for each environment
def plot():
    plot_data = open('figs/plot_data.csv', 'w')
    for env in ENVS:
        env_path = os.path.join(path, env)
        algo_folders = os.listdir(env_path)
        algo_data = None
        fig_safe, ax_safe = plt.subplots(1, 1, figsize=(30, 13), dpi=200)
        fig_reach, ax_reach = plt.subplots(1, 1, figsize=(30, 13), dpi=200)
        fig_success, ax_success = plt.subplots(1, 1, figsize=(30, 13), dpi=200)
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
            ax_safe.plot(algo_data[:, 0], algo_data[:, 5], color=ALGOS_COLOR[algo_folder], alpha=0.9,
                         label=ALGOS_LABEL[algo_folder], linestyle=ALGOS_LINESTYLE[algo_folder])
            ax_safe.fill_between(algo_data[:, 0], algo_data[:, 5] - np.sqrt(algo_data[:, 6]),
                                 np.clip(algo_data[:, 5] + np.sqrt(algo_data[:, 6]), a_min=None, a_max=100),
                                 color=ALGOS_COLOR[algo_folder], alpha=0.2)
            ax_reach.plot(algo_data[:, 0], algo_data[:, 7], color=ALGOS_COLOR[algo_folder], alpha=0.9,
                          label=ALGOS_LABEL[algo_folder], linestyle=ALGOS_LINESTYLE[algo_folder])
            ax_reach.fill_between(algo_data[:, 0], algo_data[:, 7] - np.sqrt(algo_data[:, 8]),
                                  np.clip(algo_data[:, 7] + np.sqrt(algo_data[:, 8]), a_min=None, a_max=100),
                                  color=ALGOS_COLOR[algo_folder], alpha=0.2)
            ax_success.plot(algo_data[:, 0], algo_data[:, 9], color=ALGOS_COLOR[algo_folder], alpha=0.9,
                            label=ALGOS_LABEL[algo_folder], linestyle=ALGOS_LINESTYLE[algo_folder])
            ax_success.fill_between(algo_data[:, 0], algo_data[:, 9] - np.sqrt(algo_data[:, 10]),
                                    np.clip(algo_data[:, 9] + np.sqrt(algo_data[:, 10]), a_min=None, a_max=100),
                                    color=ALGOS_COLOR[algo_folder], alpha=0.2)
        set_ax(ax_safe, env)
        ax_safe.set_ylabel('Safety rate (%)')
        set_ax(ax_reach, env)
        ax_reach.set_ylabel('Reachability rate (%)')
        set_ax(ax_success, env)
        ax_success.set_ylabel('Success rate (%)')
        fig_safe.tight_layout()
        fig_reach.tight_layout()
        fig_success.tight_layout()
        fig_safe.savefig(os.path.join('figs', f'{env}_safe.pdf'), bbox_inches='tight')
        fig_reach.savefig(os.path.join('figs', f'{env}_reach.pdf'), bbox_inches='tight')
        fig_success.savefig(os.path.join('figs', f'{env}_success.pdf'), bbox_inches='tight')
        plt.show()


if __name__ == '__main__':
    plot()
