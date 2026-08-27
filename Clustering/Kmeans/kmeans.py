import numpy as np 
import pandas as pd
from sklearn.cluster import KMeans
import jax
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


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
    'safety_label': 1  # 1 for safe
})


unsafe_data = pd.DataFrame({
    'pos_x': unsafe_df['pos_x'],
    'pos_y': unsafe_df['pos_y'],
    'h_value': unsafe_df['h_value'],
    'safety_label': 0  # 0 for unsafe
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

# Step 1: Prepare data for k-means (no scaling needed for uniform 4x4 grid)
X = df[['pos_x', 'pos_y']].values
print(f"Data shape: {X.shape}")
print(f"pos_x range: [{df['pos_x'].min():.2f}, {df['pos_x'].max():.2f}]")
print(f"pos_y range: [{df['pos_y'].min():.2f}, {df['pos_y'].max():.2f}]")

# Step 2: Silhouette analysis to find optimal k
k_range = range(2, 11)  # Try k from 2 to 10
silhouette_scores = []
inertias = []

print("\n=== Silhouette Analysis ===")
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    silhouette_avg = silhouette_score(X, labels)
    inertia = kmeans.inertia_
    
    silhouette_scores.append(silhouette_avg)
    inertias.append(inertia)
    
    print(f"k={k:2d}: silhouette={silhouette_avg:.4f}, inertia={inertia:.2f}")

# Find optimal k
optimal_k = k_range[np.argmax(silhouette_scores)]
print(f"\n✓ Optimal k: {optimal_k} (silhouette={max(silhouette_scores):.4f})")

# Step 3: Fit final k-means with optimal k
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans_final.fit_predict(X)
df['cluster'] = cluster_labels

# Step 4: Analyze clusters
print(f"\n=== Cluster Analysis (k={optimal_k}) ===")
for c in range(optimal_k):
    mask = (cluster_labels == c)
    cluster_size = mask.sum()
    agreement_rate = df[mask]['agreement'].mean()
    center = X[mask].mean(axis=0)
    
    print(f"Cluster {c}: {cluster_size:5d} points, {agreement_rate:.1%} agreement, center=({center[0]:.2f}, {center[1]:.2f})")

# Step 5: Visualization - Silhouette scores
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].plot(k_range, silhouette_scores, 'o-', linewidth=2, markersize=8, color='blue')
axes[0].axvline(x=optimal_k, color='red', linestyle='--', linewidth=2, label=f'Optimal k={optimal_k}')
axes[0].set_xlabel('k (number of clusters)', fontsize=12)
axes[0].set_ylabel('Silhouette Score', fontsize=12)
axes[0].set_title('Silhouette Analysis for Optimal k', fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=11)
axes[0].set_xticks(k_range)

axes[1].plot(k_range, inertias, 'o-', linewidth=2, markersize=8, color='green')
axes[1].axvline(x=optimal_k, color='red', linestyle='--', linewidth=2, label=f'Optimal k={optimal_k}')
axes[1].set_xlabel('k (number of clusters)', fontsize=12)
axes[1].set_ylabel('Inertia (within-cluster sum of squares)', fontsize=12)
axes[1].set_title('Elbow Method', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=11)
axes[1].set_xticks(k_range)

plt.tight_layout()
plt.savefig('silhouette_analysis.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Saved: silhouette_analysis.png")
plt.show()

# Step 6: Visualization - Position space with clusters and agreement coloring
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Left plot: Clusters with agreement coloring (black=agree, red=disagree)
colors_cluster = plt.cm.tab10(np.linspace(0, 1, optimal_k))
for c in range(optimal_k):
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
    
    # Plot cluster center
    center = kmeans_final.cluster_centers_[c]
    axes[0].scatter(center[0], center[1], c=[colors_cluster[c]], s=400, marker='*', 
                   edgecolors='black', linewidth=2, label=f'C{c} center')

axes[0].set_xlabel('pos_x', fontsize=12)
axes[0].set_ylabel('pos_y', fontsize=12)
axes[0].set_title(f'K-Means Clusters (k={optimal_k}) with Agreement Coloring\n(Black=Agreement, Red=Disagreement)', 
                 fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=9, loc='best')

# Right plot: Cluster regions colored by cluster ID
scatter = axes[1].scatter(df['pos_x'], df['pos_y'], c=cluster_labels, 
                         cmap='tab10', s=40, alpha=0.6, edgecolors='black', linewidth=0.5)
centers = kmeans_final.cluster_centers_
axes[1].scatter(centers[:, 0], centers[:, 1], c='red', s=400, marker='*', 
               edgecolors='black', linewidth=2, label='Cluster centers')
axes[1].set_xlabel('pos_x', fontsize=12)
axes[1].set_ylabel('pos_y', fontsize=12)
axes[1].set_title(f'K-Means Clusters (k={optimal_k})', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3)
plt.colorbar(scatter, ax=axes[1], label='Cluster ID')
axes[1].legend(fontsize=10)

plt.tight_layout()
plt.savefig('kmeans_clusters_analysis.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: kmeans_clusters_analysis.png")
plt.show()