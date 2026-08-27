#!/usr/bin/env python3
"""
reader.py

Robust reader for your saved graph .npz files (the format you described:
keys like "29__nodes", "29__edges", "29__senders", etc).

Features:
- Loads .npz files and groups fields by prefix (e.g. '29').
- Prints summaries for every field.
- Prints detailed node/edge feature samples for a chosen timestep.
- Exports flattened node/edge feature tables (tidy format) to .npz or .csv for fine-tuning.
- Parses `env_states` repr strings (the 'sample' string you showed) to recover numeric arrays
  for `agent`, `goal`, and `obstacle` subfields (center/width/height/theta/points) when they were
  saved as a human-readable string.

Usage:
    python reader.py path/to/episode_000000.npz

You can call functions from other scripts as well, e.g. load_npz_groups(...) then all_node_features_table(...)
"""

from collections import defaultdict
import numpy as np
import os
import sys
import re
from typing import Tuple, List, Dict, Any, Optional

# Optional: try to import GraphsTuple for "rehydration"
GraphsTupleClass = None
try:
    import jraph
    GraphsTupleClass = getattr(jraph, "GraphsTuple", None)
    print("Using jraph.GraphsTuple for rehydration.")
except Exception:
    try:
        from gcbfplus.utils.graph import GraphsTuple as ProjectGraphsTuple
        GraphsTupleClass = ProjectGraphsTuple
        print("Using gcbfplus.utils.graph.GraphsTuple for rehydration.")
    except Exception:
        GraphsTupleClass = None
        print("No GraphsTuple available; will return plain dicts for graphs.")


# -------------------------
# NPZ loading & grouping
# -------------------------
def load_npz_groups(path: str):
    """
    Load an .npz file and group fields by prefix before the first '__'.

    Returns:
      data: the loaded numpy.lib.npyio.NpzFile object
      groups: dict mapping prefix -> dict(field -> array)
      top_level: dict of top-level keys that don't contain '__'
    """
    print("Loading:", path)
    data = np.load(path, allow_pickle=True)
    print("Loaded keys:", list(data.files))

    groups = defaultdict(dict)
    top_level = {}
    for key in data.files:
        if "__" in key:
            prefix, field = key.split("__", 1)
            groups[prefix][field] = data[key]
        else:
            top_level[key] = data[key]

    # Print some top-level metadata
    print("\nTop-level keys:", list(top_level.keys()))
    if "episode" in top_level:
        try:
            print("Episode:", int(np.asarray(top_level["episode"]).item()))
        except Exception:
            print("Episode (raw):", top_level["episode"])

    return data, groups, top_level


# -------------------------
# Rehydration helper
# -------------------------
def rehydrate_group_to_graph(group_dict: Dict[str, Any]):
    """
    Try to build a GraphsTuple if available; otherwise return dict of numpy arrays.
    Accepts group_dict with keys like 'nodes','edges','senders','receivers','n_node','n_edge',
    'globals','states','env_states','node_type'.
    
    This is the key function to reconstruct full GraphsTuple from saved data.
    """
    kwargs = {}
    for key in ("nodes", "edges", "senders", "receivers", "n_node", "n_edge", "globals", "states", "env_states", "node_type"):
        if key in group_dict:
            val = group_dict[key]
            if val is None:
                kwargs[key] = None
            else:
                kwargs[key] = np.asarray(val)
    
    if not kwargs:
        return group_dict
    
    if GraphsTupleClass is not None:
        try:
            return GraphsTupleClass(**kwargs)
        except Exception as e:
            print(f"Warning: could not construct GraphsTuple ({e}); returning dict of arrays.")
            return kwargs
    else:
        return kwargs


# -------------------------
# EnvState string parsing
# -------------------------
def _find_matching_paren(s: str, start_idx: int) -> Optional[int]:
    depth = 0
    for i in range(start_idx, len(s)):
        if s[i] == "(":
            depth += 1
        elif s[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    return None


def _extract_array_text(s: str, start_pos: int) -> Optional[str]:
    """
    Given a string '...array(...' where start_pos points at the 'a' of 'array',
    return the substring inside the matching parentheses (the contents) or None.
    """
    paren_idx = s.find("(", start_pos)
    if paren_idx == -1:
        return None
    end = _find_matching_paren(s, paren_idx)
    if end is None:
        return None
    return s[paren_idx + 1:end]


def extract_array_by_key(s: str, key_name: str) -> Optional[np.ndarray]:
    """
    Search for patterns like 'key_name=array(..., dtype=...)' or 'key_name=array(...)\n'
    and evaluate the array(...) portion into a numpy array safely (only np in globals).
    Returns None if not found or on parse error.
    """
    # Find occurrences like 'agent=array('
    pat = re.compile(rf"{re.escape(key_name)}\s*=\s*array\s*\(")
    m = pat.search(s)
    if not m:
        return None
    start = m.start()
    # find substring for array contents
    arr_contents = _extract_array_text(s, s.find("array", start))
    if arr_contents is None:
        return None
    # remove potential trailing " , dtype=float32" or similar inside the contents: keep them, np.array will handle
    arr_text = "np.array(" + arr_contents + ")"
    try:
        return eval(arr_text, {"np": np}, {})
    except Exception:
        # fallback: try to strip "dtype=..." fragments (rare)
        try:
            arr_text2 = re.sub(r",\s*dtype\s*=\s*[A-Za-z0-9_]+", "", arr_text)
            return eval(arr_text2, {"np": np}, {})
        except Exception as e:
            print(f"extract_array_by_key: failed to eval {key_name}: {e}")
            return None


def extract_rectangle_fields(s: str) -> Dict[str, np.ndarray]:
    """
    If obstacle=Rectangle(...) exists in the string, extract common sub-arrays.
    Returns dict with keys possibly 'type', 'center', 'width', 'height', 'theta', 'points'.
    """
    out = {}
    rect_pat = re.compile(r"obstacle\s*=\s*Rectangle\s*\(")
    m = rect_pat.search(s)
    if not m:
        return out
    # get rectangle body
    start = s.find("(", m.start())
    end = _find_matching_paren(s, start)
    if end is None:
        return out
    rect_body = s[start + 1:end]
    for name in ("type", "center", "width", "height", "theta", "points"):
        arr = extract_array_by_key(rect_body, name)
        if arr is not None:
            out[name] = np.asarray(arr)
    return out


def parse_envstate_string(s: str) -> Dict[str, Any]:
    """
    Parse the EnvState representation string saved in env_states fields.
    Returns dict with keys: maybe 'agent', 'goal', and 'obstacle' where obstacle is a dict.
    """
    res = {}
    for key in ("agent", "goal"):
        arr = extract_array_by_key(s, key)
        if arr is not None:
            res[key] = np.asarray(arr)
    rect = extract_rectangle_fields(s)
    if rect:
        res["obstacle"] = rect
    return res


# -------------------------
# Printing / summarization
# -------------------------
def summarize_numeric(a: np.ndarray, axis=None):
    """Return (mean, std, min, max) or None if not numeric or empty."""
    a = np.asarray(a)
    if a.size == 0:
        return None
    if a.dtype.kind not in ("f", "i", "u", "b"):
        return None
    if axis is None:
        arr = a.ravel().astype(float)
        return float(arr.mean()), float(arr.std()), float(arr.min()), float(arr.max())
    else:
        arr = a.astype(float)
        return np.nanmean(arr, axis=axis), np.nanstd(arr, axis=axis), np.nanmin(arr, axis=axis), np.nanmax(arr, axis=axis)


def print_node_edge_details(group: Dict[str, Any], timestep: int = 0, show_node_count: int = 5, show_edge_count: int = 5):
    """
    Print readable node/edge features and sender/receiver mapping for one group at one timestep.
    Expects arrays in shapes:
      nodes: (T, N, F_node)
      edges: (T, E, F_edge)
      senders/receivers: (T, E)
      n_edge: (T,)
      states: maybe (T, N, F_state)
      env_states: often (1,) of strings or object reprs
    """
    # Convert to arrays safely (may be missing)
    nodes = np.asarray(group.get("nodes")) if group.get("nodes") is not None else np.array([])
    edges = np.asarray(group.get("edges")) if group.get("edges") is not None else np.array([])
    senders = np.asarray(group.get("senders")) if group.get("senders") is not None else np.array([])
    receivers = np.asarray(group.get("receivers")) if group.get("receivers") is not None else np.array([])
    n_edge = np.asarray(group.get("n_edge")) if group.get("n_edge") is not None else np.array([])
    states = group.get("states")
    env_states = group.get("env_states")

    T = nodes.shape[0] if nodes.size else 0
    if T == 0:
        print("No nodes available to inspect for this group.")
    else:
        if timestep < 0 or timestep >= T:
            print(f"Requested timestep {timestep} out of range (0..{max(0, T - 1)})")
        else:
            print(f"\n>>> Inspecting timestep {timestep} (T={T})")
            # Node features
            if nodes.size:
                try:
                    node_feat = nodes[timestep]
                except Exception:
                    # possible shape (N, F) (no time dim)
                    node_feat = nodes if nodes.ndim == 2 else nodes.reshape((-1, nodes.shape[-1]))[:show_node_count]
                print(f"Nodes: shape={node_feat.shape}, dtype={node_feat.dtype}")
                summary = summarize_numeric(node_feat)
                if summary:
                    mean, std, mn, mx = summary
                    print(f"  nodes (all): mean={mean:.4g}, std={std:.4g}, min={mn:.4g}, max={mx:.4g}")
                # per-feature summary
                if node_feat.ndim == 2:
                    perfeat = summarize_numeric(node_feat, axis=0)
                    if perfeat is not None:
                        mean_pf, std_pf, min_pf, max_pf = perfeat
                        print("  nodes per-feature (first 6 features):")
                        for i in range(min(6, node_feat.shape[1])):
                            print(f"    feat[{i}]: mean={mean_pf[i]:.4g}, std={std_pf[i]:.4g}, min={min_pf[i]:.4g}, max={max_pf[i]:.4g}")
                # sample nodes
                n_show = min(show_node_count, int(node_feat.shape[0]))
                print(f"  sample first {n_show} node vectors:")
                for i in range(n_show):
                    print(f"    node[{i}] = {np.asarray(node_feat[i]).tolist()}")

            # Edge features
            if edges.size:
                try:
                    edge_feat = edges[timestep]
                except Exception:
                    edge_feat = edges if edges.ndim == 2 else edges.reshape((-1, edges.shape[-1]))[:show_edge_count]
                print(f"\nEdges: shape={edge_feat.shape}, dtype={edge_feat.dtype}")
                summary = summarize_numeric(edge_feat)
                if summary:
                    mean, std, mn, mx = summary
                    print(f"  edges (all): mean={mean:.4g}, std={std:.4g}, min={mn:.4g}, max={mx:.4g}")
                if edge_feat.ndim == 2:
                    perfeat = summarize_numeric(edge_feat, axis=0)
                    if perfeat is not None:
                        mean_pf, std_pf, min_pf, max_pf = perfeat
                        print("  edges per-feature (first 6 features):")
                        for i in range(min(6, edge_feat.shape[1])):
                            print(f"    feat[{i}]: mean={mean_pf[i]:.4g}, std={std_pf[i]:.4g}, min={min_pf[i]:.4g}, max={max_pf[i]:.4g}")
                # sample edges + sender/receiver
                E = edge_feat.shape[0]
                send_t = senders[timestep] if senders.size else None
                recv_t = receivers[timestep] if receivers.size else None
                n_show_e = min(show_edge_count, E)
                print(f"  sample first {n_show_e} edges (sender -> receiver) and feature vector:")
                for j in range(n_show_e):
                    s = int(send_t[j]) if send_t is not None else None
                    r = int(recv_t[j]) if recv_t is not None else None
                    print(f"    edge[{j}] {s} -> {r} : {np.asarray(edge_feat[j]).tolist()}")

            # Optional: states (if present)
            if states is not None:
                st = np.asarray(states)
                if st.size:
                    try:
                        st_t = st[timestep]
                    except Exception:
                        st_t = st if st.ndim == 2 else st.reshape((-1, st.shape[-1]))[:4]
                    print(f"\nStates: shape per timestep {st_t.shape}, dtype={st_t.dtype}")
                    n_show = min(4, int(st_t.shape[0]))
                    for i in range(n_show):
                        print(f"   state[{i}] = {np.asarray(st_t[i]).tolist()}")

            # env_states: likely saved as a string repr; try to parse
            if env_states is not None:
                try:
                    # env_states may be array of length 1 with a long string
                    if isinstance(env_states, np.ndarray) and env_states.size > 0:
                        s = env_states.flat[0]
                    else:
                        s = env_states
                    if isinstance(s, bytes):
                        s = s.decode("utf8", errors="ignore")
                    if isinstance(s, str):
                        parsed = parse_envstate_string(s)
                        if parsed:
                            print("\nParsed env_states fields:")
                            for k, v in parsed.items():
                                if k == "obstacle":
                                    print("  obstacle fields:", list(v.keys()))
                                else:
                                    print(f"  {k}: shape={np.asarray(v).shape}, dtype={np.asarray(v).dtype}")
                            # show tiny samples
                            if "agent" in parsed:
                                print("  first agent row:", parsed["agent"][0].tolist())
                            if "goal" in parsed:
                                print("  first goal row:", parsed["goal"][0].tolist())
                        else:
                            print("  env_states appears to be a string but parse returned nothing.")
                except Exception as e:
                    print("  Failed to parse env_states:", e)


# -------------------------
# Flatten to tidy tables for nodes/edges (for finetuning)
# -------------------------
def all_node_features_table(group: Dict[str, Any]) -> Tuple[List[str], np.ndarray]:
    """
    Convert group['nodes'] (expected shape (T, N, F)) into a 2D table:
      rows = T * N
      columns = ['timestep', 'node_id', 'feat0', 'feat1', ...]
    Returns (header_list, table) where table shape (T*N, 2+F).
    """
    nodes = np.asarray(group.get("nodes")) if group.get("nodes") is not None else np.array([])
    if nodes.size == 0:
        return [], np.zeros((0, 0))

    # handle nodes possibly shaped (N, F) (no time dim) or (T,N,F)
    if nodes.ndim == 2:
        T = 1
        N, F = nodes.shape
        nodes_3 = nodes.reshape((1, N, F))
    elif nodes.ndim == 3:
        T, N, F = nodes.shape
        nodes_3 = nodes
    else:
        raise ValueError("expected nodes array shape (T,N,F) or (N,F), got %s" % (nodes.shape,))

    feat = nodes_3.reshape((T * N, F))
    timesteps = np.repeat(np.arange(T, dtype=np.int64), N)
    node_ids = np.tile(np.arange(N, dtype=np.int64), T)
    table = np.concatenate([timesteps[:, None], node_ids[:, None], feat.astype(np.float64)], axis=1)
    header = ["timestep", "node_id"] + [f"node_feat_{i}" for i in range(F)]
    return header, table


def all_edge_features_table(group: Dict[str, Any]) -> Tuple[List[str], np.ndarray]:
    """
    Convert edge arrays into a tidy table with sender/receiver mapping.

    Expects:
      edges shape (T, E, F_e) or (E, F_e)
      senders shape (T, E) or (E,)
      receivers shape (T, E) or (E,)
    Returns (header_list, table) with columns:
      ['timestep', 'edge_id', 'sender', 'receiver', feat0, feat1, ...]
    """
    edges = np.asarray(group.get("edges")) if group.get("edges") is not None else np.array([])
    if edges.size == 0:
        return [], np.zeros((0, 0))

    # normalize edges to (T, E, F_e)
    if edges.ndim == 2:
        T = 1
        E, F_e = edges.shape
        edges_3 = edges.reshape((1, E, F_e))
    elif edges.ndim == 3:
        T, E, F_e = edges.shape
        edges_3 = edges
    else:
        raise ValueError("expected edges array shape (T,E,F) or (E,F), got %s" % (edges.shape,))

    senders = np.asarray(group.get("senders")) if group.get("senders") is not None else None
    receivers = np.asarray(group.get("receivers")) if group.get("receivers") is not None else None

    feat = edges_3.reshape((T * E, F_e))
    timesteps = np.repeat(np.arange(T, dtype=np.int64), E)
    edge_ids = np.tile(np.arange(E, dtype=np.int64), T)

    if senders is not None:
        if senders.ndim == 1 and senders.shape[0] == E:
            send_col = np.tile(senders, T)
        elif senders.ndim == 2 and senders.shape == (T, E):
            send_col = senders.reshape(T * E)
        else:
            send_col = np.full((T * E,), -1, dtype=np.int64)
    else:
        send_col = np.full((T * E,), -1, dtype=np.int64)

    if receivers is not None:
        if receivers.ndim == 1 and receivers.shape[0] == E:
            recv_col = np.tile(receivers, T)
        elif receivers.ndim == 2 and receivers.shape == (T, E):
            recv_col = receivers.reshape(T * E)
        else:
            recv_col = np.full((T * E,), -1, dtype=np.int64)
    else:
        recv_col = np.full((T * E,), -1, dtype=np.int64)

    table = np.concatenate(
        [timesteps[:, None], edge_ids[:, None], send_col[:, None], recv_col[:, None], feat.astype(np.float64)],
        axis=1
    )
    header = ["timestep", "edge_id", "sender", "receiver"] + [f"edge_feat_{i}" for i in range(F_e)]
    return header, table


def save_table_to_npz(out_path: str, header: List[str], table: np.ndarray) -> str:
    """
    Save header + table into a .npz file.
    """
    dirname = os.path.dirname(out_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    np.savez_compressed(out_path, header=np.array(header, dtype=object), table=table)
    return out_path


def save_table_to_csv(out_path: str, header: List[str], table: np.ndarray, delimiter: str = ",") -> str:
    """
    Save header + table to CSV. This can be very large; use with caution.
    """
    dirname = os.path.dirname(out_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(out_path, "w", encoding="utf8") as f:
        f.write(delimiter.join(header) + "\n")
        for row in table:
            # Convert each value to string (preserve ints as ints)
            row_str = delimiter.join(str(x) for x in row.tolist())
            f.write(row_str + "\n")
    return out_path


# -------------------------
# Main inspector
# -------------------------
def inspect_groups(groups: Dict[str, Dict[str, Any]], rehydrate: bool = False, show_limit: int = 3):
    print("\nFound groups:", sorted(groups.keys()))
    for prefix in sorted(groups.keys()):
        print(f"\n--- Group {prefix} ---")
        g = groups[prefix]
        # list available fields
        print("fields:", list(g.keys()))
        # print dtype/shape summary for each field
        for field_name, arr in g.items():
            a = np.asarray(arr)
            print(f"  {field_name:12s} dtype={a.dtype!s:12s} shape={getattr(a, 'shape', None)}")
            if a.size > 0 and a.size <= 20:
                print("    sample:", a)
            elif a.size > 0 and a.ndim == 1 and a.size <= 200:
                print("    first10:", a.flat[:10])

        if rehydrate:
            obj = rehydrate_group_to_graph(g)
            print(f"Rehydrated object type: {type(obj)}")
            if isinstance(obj, dict):
                for kk, vv in list(obj.items())[:show_limit]:
                    print(f"   {kk}: shape={np.asarray(vv).shape} dtype={np.asarray(vv).dtype}")
            else:
                print("   GraphsTuple attrs (first few):", [a for a in dir(obj) if not a.startswith("_")][:10])

        # Print node/edge details for the first timestep (customizable)
        try:
            print_node_edge_details(g, timestep=0, show_node_count=5, show_edge_count=5)
        except Exception as e:
            print("  Failed to print node/edge details:", e)


# -------------------------
# Command-line entrypoint
# -------------------------
if __name__ == "__main__":
    path = "saved_graphs_npz/episode_000000.npz"
    if len(sys.argv) > 1:
        path = sys.argv[1]

    data, groups, top_level = load_npz_groups(path)
    # Inspect groups and print node/edge feature samples
    inspect_groups(groups, rehydrate=True)

    # Example: export nodes/edges of a target group to CSV/NPZ for finetuning
    # Change target to whichever group you want (e.g. '29' or '30')
    target = "29"
    
    if target in groups:
        # export all node features table
        header_n, table_n = all_node_features_table(groups[target])
        if header_n:
            out_npz = save_table_to_npz(f"export/{target}_nodes.npz", header_n, table_n)
            print(f"\nSaved node features table to {out_npz} (rows={table_n.shape[0]}, cols={table_n.shape[1]})")
            # save CSV optionally (comment/uncomment)
            csv_path = f"export/{target}_nodes.csv"
            save_table_to_csv(csv_path, header_n, table_n)
            print(f"Saved node features CSV to {csv_path}")

        # export all edge features table
        header_e, table_e = all_edge_features_table(groups[target])
        if header_e:
            out_npz = save_table_to_npz(f"export/{target}_edges.npz", header_e, table_e)
            print(f"Saved edge features table to {out_npz} (rows={table_e.shape[0]}, cols={table_e.shape[1]})")
            csv_path = f"export/{target}_edges.csv"
            save_table_to_csv(csv_path, header_e, table_e)
            print(f"Saved edge features CSV to {csv_path}")
    else:
        print(f"\nGroup {target} not found. Available groups: {sorted(groups.keys())}")
