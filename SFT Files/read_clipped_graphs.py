import glob
import os
from typing import Dict

import numpy as np


GRAPH_KEYS = [
    "graph_nodes",
    "graph_edges",
    "graph_states",
    "graph_senders",
    "graph_receivers",
    "graph_node_type",
    "graph_n_node",
    "graph_n_edge",
    "graph_env_states",
]

AUX_KEYS = [
    "safe_mask",
    "unsafe_mask",
    "u_qp",
    "step",
    "batch_indices",
    "graph_cost",
]

ALL_KEYS = GRAPH_KEYS + AUX_KEYS


def load_and_merge_failed_batches(data_dir: str) -> Dict[str, np.ndarray]:
    npz_files = glob.glob(os.path.join(data_dir, "*.npz"))
    if not npz_files:
        print(f"No .npz files found in {data_dir}")
        return {}

    all_samples = []

    for filepath in npz_files:
        print(f"Processing {filepath}...")
        try:
            data = np.load(filepath, allow_pickle=True)
            num_samples = int(data["num_samples"])

            for i in range(num_samples):
                prefix = f"sample_{i:04d}"
                sample = {}

                for key in GRAPH_KEYS[:-1] + ["graph_env_states", "safe_mask", "unsafe_mask", "u_qp", "step", "batch_indices"]:
                    full_key = f"{prefix}__{key}"
                    if full_key in data:
                        sample[key] = data[full_key]

                if not sample:
                    continue

                sample["graph_cost"] = np.array(50.0, dtype=np.float32)
                all_samples.append(sample)

        except Exception as e:
            print(f"Error reading {filepath}: {e}")

    if not all_samples:
        return {}

    batch_lists = {key: [] for key in ALL_KEYS}

    for sample in all_samples:
        for key, value in sample.items():
            batch_lists[key].append(value)

    num_total = len(all_samples)
    perm = np.random.permutation(num_total)
    merged = {}

    for key, lst in batch_lists.items():
        if not lst:
            continue

        try:
            stacked = np.stack(lst, axis=0)
            merged[key] = stacked[perm]
        except ValueError:
            merged[key] = np.array(lst, dtype=object)[perm]

    print(f"Loaded {num_total} total SFT samples from {len(npz_files)} files.")
    print("Assigned graph_cost = 50.0 to every NPZ graph.")
    return merged


def extract_sample(data, sample_idx: int):
    prefix = f"sample_{sample_idx:04d}"
    sample = {}

    for key in data.files:
        if key.startswith(prefix):
            field = key.replace(f"{prefix}__", "")
            sample[field] = data[key]

    sample["graph_cost"] = np.array(50.0, dtype=np.float32)
    return sample


def read_clipped_npz(filepath: str):
    data = np.load(filepath, allow_pickle=True)

    print("=" * 70)
    print(f"CLIPPED GRAPHS DATA - {filepath}")
    print("=" * 70)

    num_samples = int(data["num_samples"])
    print(f"Total samples: {num_samples}")

    if num_samples > 0:
        sample = extract_sample(data, 0)
        print("\nSample 0 keys:")
        for key in sorted(sample.keys()):
            print(f" - {key}")

        print("\nSample 0 shapes:")
        for key, value in sample.items():
            if hasattr(value, "shape"):
                print(f" - {key}: {value.shape}")
            else:
                print(f" - {key}: scalar/object")

        print(f"\nSample 0 graph_cost: {sample['graph_cost']}")

    return data


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python read_clipped_graphs.py <path_to_npz>")
        raise SystemExit(1)

    read_clipped_npz(sys.argv[1])