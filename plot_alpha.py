import matplotlib.pyplot as plt
import numpy as np
import csv
import os
import matplotlib

# data structure:
# n, epi, max_step, area_size, n_obs, safe_mean, safe_std, reach_mean, reach_std, success_mean, success_std
std_id = np.array([6, 8, 10])
path = './logs/TRO_ablations/alpha'

parameters = {
    'font.family': 'Times New Roman',
    'axes.labelsize': 48,
    'axes.titlesize': 48,
    'xtick.labelsize': 48,
    'ytick.labelsize': 48,
    'legend.fontsize': 48,
    'lines.linewidth': 4,
    'font.size': 48,
    'text.color': 'k',
    'mathtext.fontset': 'stix',
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

ALGOS_COLOR = {'gcbf+': 'C0',
               'informarl': 'C1',
               'gcbf': 'C2',
               'dec_cbf_1.0': 'C3',
               'dec_cbf_0.1': 'C4',
               'centralized_cbf_1.0': 'C5',
               'centralized_cbf_0.1': 'C6',
               }
ALGOS = ['alpha0.01',
         'alpha0.1',
         'alpha1',
         'alpha10',
         'alpha100',
         ]
#
# SEED_ALGOS = ['gcbf+', 'gcbf', 'informarl']
# ALGOS_LINESTYLE = {'gcbf+': '-',
#                    'gcbf': 'dotted',
#                    'centralized_cbf_1.0': '-.',
#                    'centralized_cbf_0.1': (5, (10, 3)),
#                    'dec_cbf_1.0': (0, (5, 1)),
#                    'dec_cbf_0.1': (0, (3, 1, 1, 1)),
#                    'informarl': '--'
#                    }

plt.rcParams.update(parameters)
bar_width = 0.12
x_labels = [r'$10^{-2}$', r'$10^{-1}$', r'$10^{0}$', r'$10^{1}$', r'$10^{2}$']
x_axis = np.arange(len(x_labels))


def set_ax(ax):
    # if env_name == 'DubinsCar':
    ax.set_xlabel(r'$\alpha$')

    # ax.set_xscale('log', base=2)
    ax.set_ylim([0, 105])
    ax.set_yticks(np.arange(0, 110, 20))
    ax.set_yticklabels(['0', '20', '40', '60', '80', '100'])
    # ax.set_xticks([8, 16, 32, 64, 128, 256, 512, 1024])
    #

    # ax.xaxis.labelpad=1000
    # ax.xaxis.set_label_coords(.9, -1)

    # Getting the current tick locations and labels
    # xticks_locs, xticks_labels = ax.get_xticks(), ax.get_xticklabels()

    # Moving the x-axis tick labels down by adjusting their y-position

    ax.tick_params(axis='x', which='both', direction='out', pad=10)
    # ax.xaxis.set_tick_params(labeltop='off', labelbottom='on', top=False, bottom=True)
    ax.set_xticks(x_axis)
    ax.set_xticklabels(x_labels)

    ax.grid(True, which='both', axis='y', linestyle='--', linewidth=2)
    # ax.set_title(env_name)


# one figure for each environment
def plot():
    plot_data = open('figs/plot_data.csv', 'w')
    # figs, axs = plt.subplots(3, 3, figsize=(34, 18), dpi=200)
    figs, ax = plt.subplots(1, 1, figsize=(20, 6.3), dpi=200)
    plt.subplots_adjust(wspace=0.125, hspace=0.1)

    algo_folders = os.listdir(path)
    algo_data = None

    idx = [ALGOS.index(algo_folder) for algo_folder in algo_folders]
    idx = np.argsort(idx)
    algo_folders = [algo_folders[i] for i in idx]

    # fig_safe, ax_safe = plt.subplots(1, 1, figsize=(30, 13), dpi=200)
    # fig_reach, ax_reach = plt.subplots(1, 1, figsize=(30, 13), dpi=200)
    # fig_success, ax_success = plt.subplots(1, 1, figsize=(30, 13), dpi=200)
    all_data = []
    for algo_folder in algo_folders:
        if algo_folder not in ALGOS:
            continue
        print(f'Processing {algo_folder}')
        algo_path = os.path.join(path, algo_folder)
        algo_data = read_data(os.path.join(algo_path, 'test_log.csv'))
        if algo_data is None:
            continue
        # for i in range(algo_data.shape[0]):
        #     plot_data.write(f'{env},{algo_folder},')
        #     for j in range(algo_data.shape[1]):
        #         plot_data.write(f'{algo_data[i, j]},')
        #     plot_data.write('\n')
        all_data.append(algo_data)

        # algo_data = algo_data[np.argsort(algo_data[:, 0])]

    all_data = np.concatenate(all_data, axis=0)

    ax.errorbar(x_axis - bar_width, all_data[:, 5], label='Safety rate',
                yerr=np.sqrt(all_data[:, 6]), ms=10, capsize=5, capthick=4, lw=4,
                color='C0', fmt='o')
    ax.errorbar(x_axis, all_data[:, 7], label='Reach rate',
                yerr=np.sqrt(all_data[:, 8]), ms=10, capsize=5, capthick=4, lw=4,
                color='C1', fmt='o')
    ax.errorbar(x_axis + bar_width, all_data[:, 9], label='Success rate',
                yerr=np.sqrt(all_data[:, 10]), ms=10, capsize=5, capthick=4, lw=4,
                color='C2', fmt='o')

    # ax.plot(x_axis - bar_width, all_data[:, 5], color='C0', lw=4)
    # ax.plot(x_axis, all_data[:, 7], color='C1', lw=4)
    # ax.plot(x_axis + bar_width, all_data[:, 9], color='C2', lw=4)


    # ax_safe.errorbar(x_axis[:algo_data.shape[0]] - 3 * bar_width + bar_width * ALGOS.index(algo_folder),
    #                  algo_data[:, 5], yerr=np.sqrt(algo_data[:, 6]), ms=10, capsize=5, capthick=4, lw=4,
    #                  fmt='o', color=ALGOS_COLOR[algo_folder], alpha=0.9, label=ALGOS_LABEL[algo_folder])
    # ax_reach.errorbar(x_axis[:algo_data.shape[0]] - 3 * bar_width + bar_width * ALGOS.index(algo_folder),
    #                   algo_data[:, 7], yerr=np.sqrt(algo_data[:, 8]), ms=10, capsize=5, capthick=4, lw=4,
    #                   fmt='o', color=ALGOS_COLOR[algo_folder], alpha=0.9, label=ALGOS_LABEL[algo_folder])
    # ax_success.errorbar(x_axis[:algo_data.shape[0]] - 3 * bar_width + bar_width * ALGOS.index(algo_folder),
    #                     algo_data[:, 9], yerr=np.sqrt(algo_data[:, 10]), ms=10, capsize=5, capthick=4, lw=4,
    #                     fmt='o', color=ALGOS_COLOR[algo_folder], alpha=0.9, label=ALGOS_LABEL[algo_folder])

    for i in range(len(x_axis)):
        ax.axvline(x=i - 0.5, color='white', linestyle='--', lw=2, alpha=0.5)
            # ax_reach.axvline(x=i - 0.5, color='white', linestyle='--', lw=2, alpha=0.5)
            # ax_success.axvline(x=i - 0.5, color='white', linestyle='--', lw=2, alpha=0.5)

    set_ax(ax)
    ax.set_ylabel('Rates (%)')
    handles, labels = ax.get_legend_handles_labels()
    handle_list, label_list = [], []
    for handle, label in zip(handles, labels):
        if label not in label_list:
            handle_list.append(handle)
            label_list.append(label)
    lgd = figs.legend(handle_list, label_list, loc='lower center', ncol=np.ceil(len(ALGOS)), bbox_to_anchor=(0.45, -0.3))

    figs.savefig(os.path.join('figs', f'alpha.pdf'), bbox_extra_artists=(lgd,), bbox_inches='tight')


if __name__ == '__main__':
    plot()