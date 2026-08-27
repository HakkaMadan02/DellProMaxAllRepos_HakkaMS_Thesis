import numpy as np 
import pandas as pd
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


try:
    safe_df = pd.read_csv('/home/ubuntu/GCBFplus_SRGasu/Probed_GCBFplus/probe/safe_agents_log.csv')
    unsafe_df = pd.read_csv('/home/ubuntu/GCBFplus_SRGasu/Probed_GCBFplus/probe/unsafe_agents_log.csv')
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit(1)


safe_data = pd.DataFrame({
    'pos_x': safe_df['pos_x'],
    'pos_y': safe_df['pos_y'],
    'h_value': safe_df['h_value'],
    'safety_label': 1
})


unsafe_data = pd.DataFrame({
    'pos_x': unsafe_df['pos_x'],
    'pos_y': unsafe_df['pos_y'],
    'h_value': unsafe_df['h_value'],
    'safety_label': 0
})


df = pd.concat([safe_data, unsafe_data], ignore_index=True)


agreement = []

for idx, row in df.iterrows():
    safety_label = row['safety_label']
    h_value = row['h_value']
    if (safety_label == 1 and h_value > 0) or (safety_label == 0 and h_value <= 0):
        agreement.append(True)
    else:
        agreement.append(False)
df['agreement'] = agreement

print(f"Total points: {len(df)}")
print(f"Agreement points: {df['agreement'].sum()}")
print(f"Disagreement points: {(~df['agreement']).sum()}")

# Extract features
X = df[['pos_x', 'pos_y']].values

# Fit GMM with n_components=2 (fixed as requested)
n_components = 2
print(f"\n=== Fitting GMM with {n_components} components ===")

gmm_final = GaussianMixture(n_components=n_components, random_state=42, n_init=10)
cluster_labels = gmm_final.fit_predict(X)
df['gmm_cluster'] = cluster_labels

# Get soft probabilities (uncertainty in cluster assignment)
probabilities = gmm_final.predict_proba(X)
df['gmm_probability'] = probabilities.max(axis=1)

# Analyze clusters
print(f"\n=== GMM Cluster Analysis (n_components={n_components}) ===")
for c in range(n_components):
    mask = (cluster_labels == c)
    cluster_size = mask.sum()
    agreement_rate = df[mask]['agreement'].mean()
    center = X[mask].mean(axis=0)

    print(f"Cluster {c}: {cluster_size:5d} points, {agreement_rate:.1%} agreement, center=({center[0]:.2f}, {center[1]:.2f})")

print(f"\n=== GMM Hyperparameters ===")
print(f"BIC: {gmm_final.bic(X):.2f}")
print(f"AIC: {gmm_final.aic(X):.2f}")
for c in range(n_components):
    print(f"Cluster {c} mean: {gmm_final.means_[c]}")
    print(f"Cluster {c} covariance:\n{gmm_final.covariances_[c]}")

# Visualization 1: GMM clusters with agreement coloring
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

colors_cluster = plt.cm.tab10(np.linspace(0, 1, n_components))

# Left plot: Agreement coloring
for c in range(n_components):
    cluster_mask = (cluster_labels == c)
    agree_in_cluster = df[cluster_mask & df['agreement']]
    disagree_in_cluster = df[cluster_mask & ~df['agreement']]

    # Plot agreement points
    if len(agree_in_cluster) > 0:
        axes[0].scatter(agree_in_cluster['pos_x'], agree_in_cluster['pos_y'],
                       c='black', s=30, alpha=0.4, edgecolors='gray', linewidth=0.5)

    # Plot disagreement points
    if len(disagree_in_cluster) > 0:
        axes[0].scatter(disagree_in_cluster['pos_x'], disagree_in_cluster['pos_y'],
                       c='red', s=50, alpha=0.7, edgecolors='darkred', linewidth=0.5)

    # Plot cluster mean
    mean = gmm_final.means_[c]
    axes[0].scatter(mean[0], mean[1], c=[colors_cluster[c]], s=400, marker='*', 
                   edgecolors='black', linewidth=2, label=f'C{c} mean')

axes[0].set_xlabel('pos_x', fontsize=12)
axes[0].set_ylabel('pos_y', fontsize=12)
axes[0].set_title(f'GMM Clusters (n={n_components}) with Agreement Coloring\n(Black=Agreement, Red=Disagreement)', 
                 fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=10, loc='best')

# Right plot: Cluster assignment with probability coloring
scatter = axes[1].scatter(df['pos_x'], df['pos_y'], c=df['gmm_probability'], 
                         cmap='viridis', s=40, alpha=0.7, edgecolors='black', linewidth=0.5)
means = gmm_final.means_
axes[1].scatter(means[:, 0], means[:, 1], c='red', s=400, marker='*', 
               edgecolors='black', linewidth=2, label='Cluster means')
cbar = plt.colorbar(scatter, ax=axes[1], label='Assignment Probability')
axes[1].set_xlabel('pos_x', fontsize=12)
axes[1].set_ylabel('pos_y', fontsize=12)
axes[1].set_title(f'GMM Clusters (n={n_components})\nColored by Assignment Confidence', 
                 fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=10)

plt.tight_layout()
plt.savefig('gmm_clusters_analysis.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Saved: gmm_clusters_analysis.png")
plt.show()

# Visualization 2: Draw covariance ellipses
fig, ax = plt.subplots(figsize=(12, 10))

# Plot all points with agreement coloring
agree_mask = df['agreement']
ax.scatter(df[agree_mask]['pos_x'], df[agree_mask]['pos_y'],
          c='black', s=20, alpha=0.3, label='Agreement', edgecolors='none')
ax.scatter(df[~agree_mask]['pos_x'], df[~agree_mask]['pos_y'],
          c='red', s=40, alpha=0.6, label='Disagreement', edgecolors='darkred', linewidth=0.5)

# Draw covariance ellipses for each component
for c in range(n_components):
    mean = gmm_final.means_[c]
    covar = gmm_final.covariances_[c]

    # Eigenvalues and eigenvectors for ellipse
    eigenvalues, eigenvectors = np.linalg.eigh(covar)
    angle = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))

    # 1-sigma and 2-sigma ellipses
    for n_std, linestyle in [(1, '--'), (2, ':')]:
        width, height = 2 * n_std * 2 * np.sqrt(eigenvalues)
        ellipse = plt.matplotlib.patches.Ellipse(mean, width, height, angle=angle, 
                                                  facecolor='none', edgecolor=colors_cluster[c],
                                                  linestyle=linestyle, linewidth=2, 
                                                  label=f'C{c} {n_std}σ')
        ax.add_patch(ellipse)

    # Plot cluster mean
    ax.scatter(mean[0], mean[1], c=[colors_cluster[c]], s=300, marker='*', 
              edgecolors='black', linewidth=2)

ax.set_xlabel('pos_x', fontsize=12)
ax.set_ylabel('pos_y', fontsize=12)
ax.set_title(f'GMM with Covariance Ellipses (n={n_components})', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10, loc='best')

plt.tight_layout()
plt.savefig('gmm_covariance_ellipses.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: gmm_covariance_ellipses.png")
plt.show()

