# graph_serial.py

import os
import numpy as np

# Try to import Jraph GraphsTuple as the canonical class; fallback names allowed
try:
    import jraph
    GraphsTupleClass = getattr(jraph, "GraphsTuple", None)
except Exception:
    GraphsTupleClass = None

# If your project defines a GraphsTuple in gcbfplus, import it if available
if GraphsTupleClass is None:
    try:
        from gcbfplus.utils.graph import GraphsTuple as ProjectGraphsTuple
        GraphsTupleClass = ProjectGraphsTuple
    except Exception:
        # leave GraphsTupleClass as None — we'll still detect objects by attr names
        GraphsTupleClass = None

def is_graphs_tuple(obj):
    """Heuristic: either instance of known class or has GraphsTuple-like attributes."""
    if obj is None:
        return False
    if GraphsTupleClass is not None and isinstance(obj, GraphsTupleClass):
        return True
    # fallback: check for the key attributes normally present on GraphsTuple objects
    required_attrs = ("nodes", "edges", "senders", "receivers", "n_edge")
    return all(hasattr(obj, a) for a in required_attrs)

def g_to_dict(g):
    """
    Convert a GraphsTuple-like object to a mapping of numpy arrays.
    Saves ALL fields needed to reconstruct GraphsTuple: nodes, edges, senders, receivers,
    n_node, n_edge, node_type, states, and env_states (as repr string for now).
    """
    if g is None:
        return None

    out = {}
    # All required fields for GraphsTuple reconstruction
    required_fields = ("nodes", "edges", "senders", "receivers", "n_node", "n_edge", "globals", "node_type", "states")
    optional_fields = ("env_states",)

    for field in required_fields + optional_fields:
        if hasattr(g, field):
            v = getattr(g, field)
            if v is None:
                out[field] = None
                continue
            
            # Special handling for env_states: save as string repr if it's a complex object
            if field == "env_states":
                try:
                    arr = np.asarray(v)
                    if arr.dtype == object or str(type(v)) not in ("ndarray", "DeviceArray"):
                        # Save as repr string
                        out[field] = np.array(repr(v), dtype=object)
                    else:
                        out[field] = arr
                except Exception:
                    out[field] = np.array(repr(v), dtype=object)
            else:
                # Convert JAX DeviceArray or lists to numpy arrays
                try:
                    arr = np.asarray(v)
                except Exception:
                    arr = np.asarray([repr(v)], dtype=object)
                out[field] = arr

    # For diagnostics: store class name
    out["_class_name"] = g.__class__.__name__
    return out

def save_episode_graphs_npz(SFT_features, episode_num, num_agents, num_obstacles,
                            out_dir="saved_graphs_npz", basename=None, compress=True):
    """
    Save all GraphsTuple-like objects found in SFT_features into a single .npz file for this episode.
    Captures complete graph information: nodes, edges, senders, receivers, n_node, n_edge,
    node_type, states, and env_states (as repr string).
    
    - SFT_features: your dict (may contain tuples like (T_graph, t) or direct GraphsTuple)
    - episode_num: int
    - out_dir: directory to store per-episode files
    Returns path to saved file.
    """
    os.makedirs(out_dir, exist_ok=True)
    if basename is None:
        basename = f"n{int(num_agents)}_obs{int(num_obstacles)}_ep{int(episode_num):02d}.npz"
    path = os.path.join(out_dir, basename)

    serial = {}
    # Always store metadata
    serial["episode"] = np.array(int(episode_num))
    serial["N"] = np.array(int(num_agents))
    serial["O"] = np.array(int(num_obstacles))

    # Walk the SFT_features dict and convert graphs to arrays
    if isinstance(SFT_features, dict):
        for k, v in SFT_features.items():
            # if it's tuple like (T_graph, t), handle graph separately
            if isinstance(v, tuple) and len(v) >= 1 and is_graphs_tuple(v[0]):
                g = v[0]
                t = v[1] if len(v) > 1 else None
                gdict = g_to_dict(g)
                # store under names that include the key, e.g. "collision_graph_nodes"
                for field_name, arr in gdict.items():
                    # skip internal classname key to avoid naming collisions
                    if field_name == "_class_name":
                        serial[f"{k}__{field_name}"] = np.array(arr if isinstance(arr, (str, bytes)) else np.asarray(arr), dtype=object) if isinstance(arr, np.ndarray) and arr.dtype == object else arr
                    else:
                        serial[f"{k}__{field_name}"] = arr
                if t is not None:
                    serial[f"{k}__timestep"] = np.array(int(t))
            # direct GraphsTuple as value
            elif is_graphs_tuple(v):
                gdict = g_to_dict(v)
                for field_name, arr in gdict.items():
                    serial[f"{k}__{field_name}"] = arr
            # list/tuple of graphs
            elif isinstance(v, (list, tuple)) and any(is_graphs_tuple(el) for el in v):
                # save each graph in the list with an index
                for i, el in enumerate(v):
                    if is_graphs_tuple(el):
                        gdict = g_to_dict(el)
                        for field_name, arr in gdict.items():
                            serial[f"{k}__{i}__{field_name}"] = arr
                    else:
                        # store non-graph entries as repr (or convertable arrays)
                        try:
                            serial[f"{k}__{i}__value"] = np.asarray(el)
                        except Exception:
                            serial[f"{k}__{i}__value_repr"] = np.array(repr(el))
            else:
                # try to store simple primitives & arrays directly; fallback to repr
                try:
                    serial[f"{k}"] = np.asarray(v)
                except Exception:
                    serial[f"{k}__repr"] = np.array(repr(v))

    else:
        # If SFT_features is not a dict, try naive storage
        try:
            serial["SFT_features"] = np.asarray(SFT_features)
        except Exception:
            serial["SFT_features__repr"] = np.array(repr(SFT_features))

    # Save as compressed npz by default (or uncompressed if compress=False)
    if compress:
        np.savez_compressed(path, **serial)
    else:
        np.savez(path, **serial)

    return path

def load_episode_graph_npz(path, rehydrate=False):
    """
    Load the .npz file saved by save_episode_graphs_npz.
    If rehydrate=True, try to reconstruct GraphsTuple objects for each graph found.
    Returns a dict mapping keys -> arrays (or GraphsTuple objects if rehydrated).
    """
    data = np.load(path, allow_pickle=True)
    out = {}
    for k in data.files:
        out[k] = data[k]
    if rehydrate:
        # Attempt to reconstruct graph dicts grouped by prefix "key__field"
        reconstructed = {}
        # group keys by prefix before the first "__"
        from collections import defaultdict
        grouped = defaultdict(dict)
        for fullkey, val in out.items():
            if "__" in fullkey:
                prefix, rest = fullkey.split("__", 1)
                grouped[prefix][rest] = val
        # For each group, try to build a GraphsTuple
        for prefix, fields in grouped.items():
            # map expected jraph names back to kwargs
            kwargs = {}
            # fields might include things like "graph__nodes", "graph__senders", etc.
            for field_name, arr in fields.items():
                if field_name in ("nodes", "edges", "senders", "receivers", "n_edge", "globals", "states", "env_states", "node_type"):
                    kwargs[field_name] = np.asarray(arr)
            # try to construct GraphsTuple if possible
            if kwargs:
                try:
                    if GraphsTupleClass is not None:
                        # NOTE: different GraphsTuple implementations might expect different arg names.
                        gt = GraphsTupleClass(**kwargs)
                    else:
                        # Build a simple lightweight dict if we can't import GraphsTuple
                        gt = kwargs
                except Exception:
                    gt = kwargs
                reconstructed[prefix] = gt
        # Merge reconstructed graphs into out under keys like out['graph'] = gt
        out_rehydrated = dict(out)
        out_rehydrated.update(reconstructed)
        return out_rehydrated

    return out

