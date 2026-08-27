import pickle
import sys
import os

# --- ADD THIS IMPORT ---
# You need to point to where GraphsTuple is defined. 
# Based on your test.py, it comes from 'gcbfplus.utils.graph'
from gcbfplus.utils.graph import GraphsTuple 
# -----------------------

def load_all_data(file_path):
    with open(file_path, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break

if __name__ == "__main__":
    file_path = "safety_features_SFT_log.pkl"
    
    # Verify file exists
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        sys.exit(1)

    all_episodes = list(load_all_data(file_path))
    print(f"Total episodes found: {len(all_episodes)}")

    if all_episodes:
        print("First episode keys:", all_episodes[0].keys())
        # Example: Accessing the GraphsTuple inside the data
        # print(all_episodes[0]['safety_features_SFT']['some_key'])
