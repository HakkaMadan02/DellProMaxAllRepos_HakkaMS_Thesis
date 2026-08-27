"""
Quick script to read and inspect batch data
"""
import numpy as np

# Load the batch data
data = np.load('batch_step_0.npz')

print("="*70)
print("BATCH DATA - Step 0")
print("="*70)

print("\n📋 Available keys:")
for key in data.keys():
    print(f"  - {key}")

print("\n📊 Data shapes and info:")
print(f"  Step: {data['step']}")
print(f"  Batch indices shape: {data['batch_indices'].shape}")
print(f"  Graph nodes shape: {data['graph_nodes'].shape}")
print(f"  Graph edges shape: {data['graph_edges'].shape}")
print(f"  Graph states shape: {data['graph_states'].shape}")
print(f"  Safe mask shape: {data['safe_mask'].shape}")
print(f"  Unsafe mask shape: {data['unsafe_mask'].shape}")
print(f"  QP actions shape: {data['u_qp'].shape}")

print("\n🎭 Mask statistics:")
safe_count = np.sum(data['safe_mask'])
unsafe_count = np.sum(data['unsafe_mask'])
total = data['safe_mask'].size
print(f"  Safe samples: {safe_count}/{total} ({100*safe_count/total:.1f}%)")
print(f"  Unsafe samples: {unsafe_count}/{total} ({100*unsafe_count/total:.1f}%)")

print("\n📍 First 3 samples:")
for i in range(min(3, len(data['graph_states']))):
    print(f"\n  Sample {i}:")
    print(f"    State: {data['graph_states'][i]}")
    print(f"    Safe: {data['safe_mask'][i]}, Unsafe: {data['unsafe_mask'][i]}")
    print(f"    Action: {data['u_qp'][i]}")

print("\n" + "="*70)
print("✅ Done! Access data with: data['key_name']")
print("="*70)
