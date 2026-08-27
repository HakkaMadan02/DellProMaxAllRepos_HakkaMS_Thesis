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
    try:
        with open(path_to_csv, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                data.append(np.array(row, dtype=np.float32))
                data[-1][std_id] = data[-1][std_id] ** 2  # we store variance instead of std
    except FileNotFoundError:
        print(f'Warning: cannot find data in {path_to_csv}')
        return None
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


ENVS = ['LinearDrone', 'CFHL']
ALGOS = ['gcbf+',
         'informarl',
         'gcbf',
         'mpc',
         'dec_cbf_1.0',
         'dec_cbf_0.1',
         'centralized_cbf_1.0',
         'centralized_cbf_0.1',
         ]
ALGOS_COLOR = {'gcbf+': 'C0',
               'informarl': 'C1',
               'gcbf': 'C2',
               'dec_cbf_1.0': 'C3',
               'dec_cbf_0.1': 'C4',
               'centralized_cbf_1.0': 'C5',
               'centralized_cbf_0.1': 'C6',
               'mpc': 'C7'
               }
ENV_COLOR = {'SingleIntegrator': 'C0',
             'DoubleIntegrator': 'C1',
             'DubinsCar': 'C2'}
ALGOS_LABEL = {'gcbf+': 'GCBF+',
               'informarl': 'InforMARL',
               'gcbf': 'GCBFv0',
               'mpc': 'MPC',
               'dec_cbf_1.0': 'DecCBF1.0',
               'dec_cbf_0.1': 'DecCBF0.1',
               'centralized_cbf_1.0': 'CBF1.0',
               'centralized_cbf_0.1': 'CBF0.1',
               }
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
bar_width = 0.1
x_labels = ['8', '16', '32', '64', '128', '256', '512', '1024']
x_axis = np.arange(len(x_labels))


def set_ax(ax, env_name):
    if env_name == 'CFHL':
        ax.set_xlabel('Number of agents')

    # ax.set_xscale('log', base=2)
    ax.set_ylim([0, 105])
    ax.set_yticks(np.arange(0, 110, 20))
    ax.set_yticklabels(['0', '20', '40', '60', '80', '100'])
    # ax.set_xticks([8, 16, 32, 64, 128, 256, 512, 1024])
    ax.set_xticks(x_axis)
    if env_name == 'CFHL':
        # ax.set_xticklabels(['8', '16', '32', '64', '128', '256', '512', '1024'], rotation=30)
        ax.set_xticklabels(['8', '16', '32', '64', '128', '256', '512', '1024'])
    else:
        ax.get_xaxis().set_ticklabels([])
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(True, which='both', axis='y', linestyle='--', linewidth=2)
    # ax.set_title(env_name)


# one figure for each environment
def plot():
    plot_data = open('figs/plot_data.csv', 'w')
    # figs, axs = plt.subplots(3, 3, figsize=(34, 18), dpi=200)
    figs, axs = plt.subplots(2, 3, figsize=(60, 13.1), dpi=200)
    plt.subplots_adjust(wspace=0.125, hspace=0.1)
    for env in ENVS:
        # envs are columns
        ax_success = axs[ENVS.index(env), 0]
        ax_safe = axs[ENVS.index(env), 1]
        ax_reach = axs[ENVS.index(env), 2]

        env_path = os.path.join(path, env)
        algo_folders = os.listdir(env_path)
        algo_data = None

        idx = [ALGOS.index(algo_folder) for algo_folder in algo_folders]
        idx = np.argsort(idx)
        algo_folders = [algo_folders[i] for i in idx]

        # fig_safe, ax_safe = plt.subplots(1, 1, figsize=(30, 13), dpi=200)
        # fig_reach, ax_reach = plt.subplots(1, 1, figsize=(30, 13), dpi=200)
        # fig_success, ax_success = plt.subplots(1, 1, figsize=(30, 13), dpi=200)
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
                             algo_data[:, 5], yerr=np.sqrt(algo_data[:, 6]), ms=10, capsize=5, capthick=4, lw=4,
                             fmt='o', color=ALGOS_COLOR[algo_folder], alpha=0.9, label=ALGOS_LABEL[algo_folder])
            ax_reach.errorbar(x_axis[:algo_data.shape[0]] - 3 * bar_width + bar_width * ALGOS.index(algo_folder),
                              algo_data[:, 7], yerr=np.sqrt(algo_data[:, 8]), ms=10, capsize=5, capthick=4, lw=4,
                              fmt='o', color=ALGOS_COLOR[algo_folder], alpha=0.9, label=ALGOS_LABEL[algo_folder])
            ax_success.errorbar(x_axis[:algo_data.shape[0]] - 3 * bar_width + bar_width * ALGOS.index(algo_folder),
                                algo_data[:, 9], yerr=np.sqrt(algo_data[:, 10]), ms=10, capsize=5, capthick=4, lw=4,
                                fmt='o', color=ALGOS_COLOR[algo_folder], alpha=0.9, label=ALGOS_LABEL[algo_folder])

            for i in range(len(x_axis)):
                ax_safe.axvline(x=i - 0.5, color='white', linestyle='--', lw=2, alpha=0.5)
                ax_reach.axvline(x=i - 0.5, color='white', linestyle='--', lw=2, alpha=0.5)
                ax_success.axvline(x=i - 0.5, color='white', linestyle='--', lw=2, alpha=0.5)
            # ax_safe.axvline(x=x_axis - 0.5, color='k', linestyle='--', lw=4)

            # ax_safe.plot(x_axis[:algo_data.shape[0]] - 3.5 * bar_width + bar_width * ALGOS.index(algo_folder),
            #              algo_data[:, 5], color=ALGOS_COLOR[algo_folder], alpha=0.9, lw=4)
            # ax_reach.plot(x_axis[:algo_data.shape[0]] - 3.5 * bar_width + bar_width * ALGOS.index(algo_folder),
            #                 algo_data[:, 7], color=ALGOS_COLOR[algo_folder], alpha=0.9, lw=4)
            # ax_success.plot(x_axis[:algo_data.shape[0]] - 3.5 * bar_width + bar_width * ALGOS.index(algo_folder),
            #                 algo_data[:, 9], color=ALGOS_COLOR[algo_folder], alpha=0.9, lw=4)

            # ax_safe.bar(x_axis[:algo_data.shape[0]] - 3.5 * bar_width + bar_width * ALGOS.index(algo_folder),
            #             algo_data[:, 5], width=bar_width,
            #             color=ALGOS_COLOR[algo_folder], alpha=0.9, label=ALGOS_LABEL[algo_folder],
            #             yerr=np.sqrt(algo_data[:, 6]), error_kw=dict(lw=1, capsize=3, capthick=2))
            # ax_reach.bar(x_axis[:algo_data.shape[0]] - 3.5 * bar_width + bar_width * ALGOS.index(algo_folder),
            #              algo_data[:, 7], width=bar_width,
            #              color=ALGOS_COLOR[algo_folder], alpha=0.9, label=ALGOS_LABEL[algo_folder],
            #              yerr=np.sqrt(algo_data[:, 8]), error_kw=dict(lw=1, capsize=3, capthick=2))
            # ax_success.bar(x_axis[:algo_data.shape[0]] - 3.5 * bar_width + bar_width * ALGOS.index(algo_folder),
            #                algo_data[:, 9], width=bar_width,
            #                color=ALGOS_COLOR[algo_folder], alpha=0.9, label=ALGOS_LABEL[algo_folder],
            #                yerr=np.sqrt(algo_data[:, 10]), error_kw=dict(lw=1, capsize=3, capthick=2))

            # ax_safe.plot(algo_data[:, 0], algo_data[:, 5], color=ALGOS_COLOR[algo_folder], alpha=0.9,
            #              label=ALGOS_LABEL[algo_folder], linestyle=ALGOS_LINESTYLE[algo_folder])
            # ax_safe.fill_between(algo_data[:, 0], algo_data[:, 5] - np.sqrt(algo_data[:, 6]),
            #                      np.clip(algo_data[:, 5] + np.sqrt(algo_data[:, 6]), a_min=None, a_max=100),
            #                      color=ALGOS_COLOR[algo_folder], alpha=0.2)
            # ax_reach.plot(algo_data[:, 0], algo_data[:, 7], color=ALGOS_COLOR[algo_folder], alpha=0.9,
            #               label=ALGOS_LABEL[algo_folder], linestyle=ALGOS_LINESTYLE[algo_folder])
            # ax_reach.fill_between(algo_data[:, 0], algo_data[:, 7] - np.sqrt(algo_data[:, 8]),
            #                       np.clip(algo_data[:, 7] + np.sqrt(algo_data[:, 8]), a_min=None, a_max=100),
            #                       color=ALGOS_COLOR[algo_folder], alpha=0.2)
            # ax_success.plot(algo_data[:, 0], algo_data[:, 9], color=ALGOS_COLOR[algo_folder], alpha=0.9,
            #                 label=ALGOS_LABEL[algo_folder], linestyle=ALGOS_LINESTYLE[algo_folder])
            # ax_success.fill_between(algo_data[:, 0], algo_data[:, 9] - np.sqrt(algo_data[:, 10]),
            #                         np.clip(algo_data[:, 9] + np.sqrt(algo_data[:, 10]), a_min=None, a_max=100),
            #                         color=ALGOS_COLOR[algo_folder], alpha=0.2)
        set_ax(ax_safe, env)
        ax_safe.set_ylabel('Safety rate (%)')
        set_ax(ax_reach, env)
        ax_reach.set_ylabel('Reach rate (%)')
        set_ax(ax_success, env)
        ax_success.set_ylabel('Success rate (%)')
    handles, labels = axs[0, 0].get_legend_handles_labels()
    label_list = list(ALGOS_LABEL.values())
    idx = [label_list.index(label) for label in labels]
    idx = np.argsort(idx)
    # idx = [ALGOS_LABEL.index(label) for label in labels]
    # idx = np.argsort(labels)
    handles = [handles[i] for i in idx]
    labels = [labels[i] for i in idx]
    lgd = figs.legend(handles, labels, loc='lower center', ncol=np.ceil(len(ALGOS)), bbox_to_anchor=(0.48, -0.1))
    # figs.tight_layout()
    # text0 = figs.text(0.05, 0.77, 'Single\nIntegrator', ha='center', va='center')
    # text1 = figs.text(0.05, 0.495, 'Double\nIntegrator', ha='center', va='center')
    # text2 = figs.text(0.05, 0.21, 'Dubins\nCar', ha='center', va='center')

    text_gap = 0.41
    text_low = 0.28

    # text0 = figs.text(0.08, text_low + 2 * text_gap, 'Single\nIntegrator', ha='center', va='center', rotation=90)
    text1 = figs.text(0.08, text_low + text_gap, 'Linear\nDrone', ha='center', va='center', rotation=90)
    text2 = figs.text(0.08, text_low, 'Crazyflie\nDrone', ha='center', va='center', rotation=90)

    figs.patches.extend([plt.Rectangle((-0.05, -1.0),
                                       1.0, 1.0 + text_low + 0.5 * text_gap,
                                       fill=False, color='C7', alpha=0.2, zorder=-1,
                                       transform=figs.transFigure, figure=figs)])
    # figs.patches.extend([plt.Rectangle((-0.05, -1.0),
    #                                    1.0, 1.0 + text_low + 1.5 * text_gap,
    #                                    fill=False, color='C7', alpha=0.2, zorder=-1,
    #                                    transform=figs.transFigure, figure=figs)])

    # figs.patches.extend([plt.Rectangle((-0.05, -1.0),
    #                                    1.0, 1.36,
    #                                    fill=False, color='C7', alpha=0.2, zorder=-1,
    #                                    transform=figs.transFigure, figure=figs)])
    # figs.patches.extend([plt.Rectangle((-0.05, 0.355),
    #                                    1.0, 0.265,
    #                                    fill=False, color='C7', alpha=0.5, zorder=-1,
    #                                    transform=figs.transFigure, figure=figs, linewidth=2)])
    # figs.patches.extend([plt.Rectangle((-0.05, 0.62),
    #                                    1.0, 0.3,
    #                                    fill=False, color='C7', alpha=0.5, zorder=-1,
    #                                    transform=figs.transFigure, figure=figs, linewidth=2)])



    # for env in ENVS:
    #     bbox0 = axs[ENVS.index(env), 0].get_tightbbox()
    #     bbox1 = axs[ENVS.index(env), 2].get_tightbbox()
    #     figs.patches.extend([plt.Rectangle((bbox0.x0, bbox0.y0),
    #                                        # bbox1.x1 - bbox0.x0, bbox1.y1 - bbox0.y0,
    #                                        bbox0.width, bbox0.height,
    #                                        fill=True, color=ENV_COLOR[env], alpha=0.2, zorder=-1,
    #                                        transform=None, figure=figs)])
    #
    # figs.patches.extend([plt.Rectangle((bbox.x0 - pad_h / 2, bbox.y0 - pad_v / 2),
    #                                    bbox.width + pad_h, bbox.height + pad_v,
    #                                    fill=True, color=ENV_COLOR[env], alpha=0.2, zorder=-1,
    #                                    transform=None, figure=figs)])

    figs.savefig(os.path.join('figs', f'env3d.pdf'), bbox_extra_artists=(lgd, text1, text2), bbox_inches='tight')
    # figs.savefig(os.path.join('figs', f'{env}_safe.pdf'), bbox_inches='tight')
    # figs.savefig(os.path.join('figs', f'{env}_reach.pdf'), bbox_inches='tight')
    # figs.savefig(os.path.join('figs', f'{env}_success.pdf'), bbox_inches='tight')
    # plt.show()


if __name__ == '__main__':
    plot()
