import numpy as np 
import pandas as pd
from sklearn.cluster import KMeans
import jax
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


try:
    safe_df = pd.read_csv('/home/ubuntu/GCBFplus_SRGasu/Probed_GCBFplus/probe/test_smallPOP_safe.csv')
    unsafe_df = pd.read_csv('/home/ubuntu/GCBFplus_SRGasu/Probed_GCBFplus/probe/test_smallPOP_unsafe.csv')
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

# Filter only disagreement points
df_disagree = df[~df['agreement']].copy()
print(f"\n✓ Filtered to disagreement points: {len(df_disagree)}")

# Extract features ONLY from disagreement data
X_disagree = df_disagree[['pos_x', 'pos_y']].values

# Silhouette analysis on disagreement points
k_range = range(2, 10)
silhouette_scores = []
inertias = []

print("\n=== Silhouette Analysis (Disagreement Points Only) ===")
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_disagree)
    silhouette_avg = silhouette_score(X_disagree, cluster_labels)
    inertia = kmeans.inertia_
    
    silhouette_scores.append(silhouette_avg)
    inertias.append(inertia)
    print(f"k={k:2d}: silhouette={silhouette_avg:.4f}, inertia={inertia:.2f}")

# Find optimal k from disagreement data
optimal_k = k_range[np.argmax(silhouette_scores)]
print(f"\n✓ Optimal k (disagreement): {optimal_k} (silhouette={max(silhouette_scores):.4f})")

# Fit final k-means on disagreement points
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels_disagree = kmeans_final.fit_predict(X_disagree)
df_disagree['cluster'] = cluster_labels_disagree

# Analyze disagreement clusters
print(f"\n=== Cluster Analysis (k={optimal_k}, Disagreement Points) ===")
for c in range(optimal_k):
    mask = (cluster_labels_disagree == c)
    cluster_size = mask.sum()
    center = X_disagree[mask].mean(axis=0)
    
    print(f"Cluster {c}: {cluster_size:5d} disagreement points, center=({center[0]:.2f}, {center[1]:.2f})")

# Visualization 1: Silhouette analysis
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].plot(k_range, silhouette_scores, 'o-', linewidth=2, markersize=8, color='blue')
axes[0].axvline(x=optimal_k, color='red', linestyle='--', linewidth=2, label=f'Optimal k={optimal_k}')
axes[0].set_xlabel('k (number of clusters)', fontsize=12)
axes[0].set_ylabel('Silhouette Score', fontsize=12)
axes[0].set_title('Silhouette Analysis for Optimal k\n(Disagreement Points Only)', fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=11)
axes[0].set_xticks(k_range)

axes[1].plot(k_range, inertias, 'o-', linewidth=2, markersize=8, color='green')
axes[1].axvline(x=optimal_k, color='red', linestyle='--', linewidth=2, label=f'Optimal k={optimal_k}')
axes[1].set_xlabel('k (number of clusters)', fontsize=12)
axes[1].set_ylabel('Inertia (within-cluster sum of squares)', fontsize=12)
axes[1].set_title('Elbow Method\n(Disagreement Points Only)', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=11)
axes[1].set_xticks(k_range)

plt.tight_layout()
plt.savefig('silhouette_analysis_disagreement.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Saved: silhouette_analysis_disagreement.png")
plt.show()

# Visualization 2: Disagreement clusters in position space
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Left plot: Only disagreement points, colored by cluster
colors_cluster = plt.cm.tab10(np.linspace(0, 1, optimal_k))
for c in range(optimal_k):
    cluster_mask = (cluster_labels_disagree == c)
    axes[0].scatter(df_disagree[cluster_mask]['pos_x'], 
                   df_disagree[cluster_mask]['pos_y'],
                   c=[colors_cluster[c]], s=60, alpha=0.7, label=f'Cluster {c}',
                   edgecolors='black', linewidth=0.5)
    
    # Plot cluster center
    center = kmeans_final.cluster_centers_[c]
    axes[0].scatter(center[0], center[1], c=[colors_cluster[c]], s=400, marker='*', 
                   edgecolors='black', linewidth=2)

axes[0].set_xlabel('pos_x', fontsize=12)
axes[0].set_ylabel('pos_y', fontsize=12)
axes[0].set_title(f'K-Means Clusters (k={optimal_k})\nDisagreement Points Only', fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=10, loc='best')

# Right plot: Full data context - agreement (black) vs disagreement (colored by cluster)
# Plot agreement points in background
agree_mask = df['agreement']
axes[1].scatter(df[agree_mask]['pos_x'], df[agree_mask]['pos_y'],
               c='lightgray', s=20, alpha=0.3, label='Agreement', edgecolors='none')

# Plot disagreement points colored by cluster
for c in range(optimal_k):
    cluster_mask = (cluster_labels_disagree == c)
    axes[1].scatter(df_disagree[cluster_mask]['pos_x'], 
                   df_disagree[cluster_mask]['pos_y'],
                   c=[colors_cluster[c]], s=60, alpha=0.8, label=f'Disagree Cluster {c}',
                   edgecolors='black', linewidth=0.5)
    
    # Plot cluster center
    center = kmeans_final.cluster_centers_[c]
    axes[1].scatter(center[0], center[1], c=[colors_cluster[c]], s=400, marker='*', 
                   edgecolors='black', linewidth=2)

axes[1].set_xlabel('pos_x', fontsize=12)
axes[1].set_ylabel('pos_y', fontsize=12)
axes[1].set_title(f'K-Means Clusters of Disagreement\n(Gray background = Agreement)', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=9, loc='best')

plt.tight_layout()
plt.savefig('kmeans_disagreement_clusters.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: kmeans_disagreement_clusters.png")
plt.show()