import numpy as np 
import pandas as pd
from sklearn.cluster import KMeans
import jax
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


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


print(f"Agreement rate: {sum(agreement) / len(agreement):.1%}")
print(f"Disagreement rate: {(1 - sum(agreement) / len(agreement)):.1%}")
print(f"Total agreement cases: {sum(agreement)}")
print(f"Total disagreement cases: {len(agreement) - sum(agreement)}")

# Step 1: Visualization - Position space with agreement coloring
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Left plot: Black (agreement) vs Red (disagreement)
agree_mask = df['agreement']
axes[0].scatter(df[agree_mask]['pos_x'], 
               df[agree_mask]['pos_y'],
               c='black', s=30, alpha=0.4, label='Agreement', edgecolors='gray', linewidth=0.5)
axes[0].scatter(df[~agree_mask]['pos_x'], 
               df[~agree_mask]['pos_y'],
               c='red', s=50, alpha=0.7, label='Disagreement', edgecolors='darkred', linewidth=0.5)
axes[0].set_xlabel('pos_x', fontsize=12)
axes[0].set_ylabel('pos_y', fontsize=12)
axes[0].set_title('Safety Label vs CBF Prediction (Black=Agree, Red=Disagree)', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Right plot: Ground truth labels (green=safe, orange=unsafe)
safe_mask = df['safety_label'] == 1
axes[1].scatter(df[safe_mask]['pos_x'], 
               df[safe_mask]['pos_y'],
               c='green', s=30, alpha=0.5, label='Safe (ground truth)', edgecolors='darkgreen', linewidth=0.5)
axes[1].scatter(df[~safe_mask]['pos_x'], 
               df[~safe_mask]['pos_y'],
               c='orange', s=30, alpha=0.5, label='Unsafe (ground truth)', edgecolors='darkorange', linewidth=0.5)
axes[1].set_xlabel('pos_x', fontsize=12)
axes[1].set_ylabel('pos_y', fontsize=12)
axes[1].set_title('Ground Truth Safety Labels', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('discrepancy_analysis_position_space.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: discrepancy_analysis_position_space.png")
plt.show()

# Step 2: CBF value distribution analysis
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Left: h_value distribution by ground truth
safe_h = df[df['safety_label'] == 1]['h_value']
unsafe_h = df[df['safety_label'] == 0]['h_value']
axes[0].hist(safe_h, bins=60, alpha=0.6, label='Safe', color='green', edgecolor='darkgreen')
axes[0].hist(unsafe_h, bins=60, alpha=0.6, label='Unsafe', color='orange', edgecolor='darkorange')
axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2, label='h=0 boundary')
axes[0].set_xlabel('h_value (GCBF+ prediction)', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_title('CBF Distribution by Ground Truth', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3, axis='y')

# Right: h_value distribution by agreement
agree_h = df[df['agreement']]['h_value']
disagree_h = df[~df['agreement']]['h_value']
axes[1].hist(agree_h, bins=60, alpha=0.6, label='Agreement', color='black', edgecolor='gray')
axes[1].hist(disagree_h, bins=60, alpha=0.6, label='Disagreement', color='red', edgecolor='darkred')
axes[1].axvline(x=0, color='blue', linestyle='--', linewidth=2, label='h=0 boundary')
axes[1].set_xlabel('h_value (GCBF+ prediction)', fontsize=12)
axes[1].set_ylabel('Count', fontsize=12)
axes[1].set_title('CBF Distribution by Agreement Status', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('cbf_distribution_analysis.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: cbf_distribution_analysis.png")
plt.show()

