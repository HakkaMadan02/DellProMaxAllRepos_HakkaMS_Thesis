import csv
import matplotlib.pyplot as plt
import os
plt.style.use('seaborn-v0_8-dark')
n_agents = 3

senders = []
with open('send.csv', "r") as f:
    reader = csv.reader(f,  delimiter=',')
    for i in reader:
        for j in i:
            if j != ' ':
                senders.append(int(j))

receivers = []
with open('recv.csv', "r") as f:
    reader = csv.reader(f,  delimiter=',')
    for i in reader:
        for j in i:
            if j != ' ':
                receivers.append(int(j))

dists = []
with open('dist.csv', "r") as f:
    reader = csv.reader(f,  delimiter=',')
    for i in reader:
        for j in i:
            if j != ' ':
                dists.append(float(j))

attns = []
with open('attn.csv', "r") as f:
    reader = csv.reader(f,  delimiter=',')
    for i in reader:
        for j in i:
            if j != ' ':
                attns.append(float(j))

plot_attn = []
plot_dist = []
for i in range(len(senders)):
    if senders[i] < n_agents:
    # if senders[i] == 2:
    #     if receivers[i] == 0:
        plot_attn.append(attns[i])
        plot_dist.append(dists[i])
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
fig, ax = plt.subplots(1, 1, figsize=(10, 8), dpi=300)

# plt.plot(plot_attn)
for i in range(len(plot_attn)):
    plt.scatter(plot_dist[i], plot_attn[i], c='C3')
    # plt.plot()
plt.axvline(x=0.5, color='k', lw=3)
ax.text(0.495, 0.2, r'$R=0.5$', ha='right')
ax.set_xticks([0.2, 0.3, 0.4, 0.5])
ax.set_xticklabels(['0.2', '0.3', '0.4', '0.5'])
ax.set_yticks([0.0, 0.1, 0.2, 0.3])
ax.set_yticklabels(['0.0', '0.1', '0.2', '0.3'])
ax.set_xlabel(r'Inter-agent distance $d_{ij}$')
ax.set_ylabel(r'Attention weight $w_{ij}$')
plt.tight_layout()
plt.savefig(os.path.join('figs', f'attn-dist.pdf'), bbox_inches='tight')
