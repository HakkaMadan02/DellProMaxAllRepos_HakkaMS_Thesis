"""
Metric summary script — GCBF+ only (Pre-Trained vs. Attention-Repair)
No plotting, no file output — terminal prints only.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# -----------------------------
# CSV file paths (edit as needed)
# -----------------------------
AGENT_COUNTS = ["8", "16", "24", "32"]

PRETRAINED_FILES = {
    "8":  "episode_metrics_OG_8.csv",
    "16": "episode_metrics_OG_16.csv",
    "24": "episode_metrics_OG_24.csv",
    "32": "episode_metrics_OG_32.csv",
}

ATTENTION_FILES = {
    "8":  "episode_metrics_Attention_8.csv",
    "16": "episode_metrics_Attention_16.csv",
    "24": "episode_metrics_Attention_24.csv",
    "32": "episode_metrics_Attention_32.csv",
}

FINISH_COL = "finish_rate"          # stored as 0-1 fraction
UNSAFE_COL = "labeled_unsafe_count" # absolute counts


# -----------------------------
# Helpers
# -----------------------------
def load_column(csv_path: str, column_name: str) -> np.ndarray:
    df = pd.read_csv(csv_path)
    normalized = {c.strip().lower(): c for c in df.columns}
    if column_name.lower() not in normalized:
        raise ValueError(
            f"Column '{column_name}' not found in {csv_path}. "
            f"Available columns: {list(df.columns)}"
        )
    actual_col = normalized[column_name.lower()]
    values = df[actual_col].dropna().astype(float).to_numpy()
    if len(values) != 32:
        print(f"  [warn] {csv_path}: {len(values)} rows in '{actual_col}', expected 32.")
    return values


def summarize(values: np.ndarray) -> tuple[float, float, float]:
    """Returns (mean, std[population, ddof=0], total/sum)."""
    return float(values.mean()), float(values.std(ddof=0)), float(values.sum())


# -----------------------------
# Load & compute per-file stats
# -----------------------------
def main():
    finish_stats = {"Pre-Trained": {}, "Attention": {}}
    unsafe_stats = {"Pre-Trained": {}, "Attention": {}}

    file_maps = {"Pre-Trained": PRETRAINED_FILES, "Attention": ATTENTION_FILES}

    for variant, files in file_maps.items():
        for n_agents in AGENT_COUNTS:
            path = files[n_agents]

            finish_vals = load_column(path, FINISH_COL)
            f_mean, f_std, f_total = summarize(finish_vals)
            finish_stats[variant][n_agents] = {"mean": f_mean, "std": f_std}

            unsafe_vals = load_column(path, UNSAFE_COL)
            u_mean, u_std, u_total = summarize(unsafe_vals)
            unsafe_stats[variant][n_agents] = {"mean": u_mean, "std": u_std, "total": u_total}

    # -----------------------------
    # 1. Finish rate report
    # -----------------------------
    print("\n" + "=" * 60)
    print("1. FINISH RATE (mean ± std, fraction 0-1)")
    print("=" * 60)
    for variant in ["Pre-Trained", "Attention"]:
        for n_agents in AGENT_COUNTS:
            s = finish_stats[variant][n_agents]
            print(f"  {variant:12s} | agents={n_agents:>3s} | "
                  f"mean={s['mean']:.4f} ({s['mean']*100:.2f}%)  std={s['std']:.4f}")

    # -----------------------------
    # 2. Unsafe state report
    # -----------------------------
    print("\n" + "=" * 60)
    print("2. LABELED UNSAFE STATES (mean ± std, absolute count)")
    print("=" * 60)
    for variant in ["Pre-Trained", "Attention"]:
        for n_agents in AGENT_COUNTS:
            s = unsafe_stats[variant][n_agents]
            print(f"  {variant:12s} | agents={n_agents:>3s} | "
                  f"mean={s['mean']:.2f}  std={s['std']:.2f}  total={s['total']:.0f}")

    # -----------------------------
    # 3. Trade-off summary
    #    Safety   -> uses ABSOLUTE TOTALS (sum over episodes)
    #    Finish   -> uses MEANS
    # -----------------------------
    print("\n" + "=" * 60)
    print("3. TRADE-OFF: Pre-Trained -> Attention (agent-wise)")
    print("=" * 60)
    for n_agents in AGENT_COUNTS:
        pre_unsafe_total = unsafe_stats["Pre-Trained"][n_agents]["total"]
        att_unsafe_total = unsafe_stats["Attention"][n_agents]["total"]
        safety_pct_reduction = (
            (pre_unsafe_total - att_unsafe_total) / pre_unsafe_total * 100.0
            if pre_unsafe_total != 0 else float("nan")
        )

        pre_finish_mean = finish_stats["Pre-Trained"][n_agents]["mean"]
        att_finish_mean = finish_stats["Attention"][n_agents]["mean"]
        finish_pct_decline = (
            (pre_finish_mean - att_finish_mean) / pre_finish_mean * 100.0
            if pre_finish_mean != 0 else float("nan")
        )

        print(f"\n  Agents = {n_agents}")
        print(f"    Unsafe states   : Pre-Trained total={pre_unsafe_total:.0f}  "
              f"Attention total={att_unsafe_total:.0f}  "
              f"=> % reduction = {safety_pct_reduction:+.2f}%")
        print(f"    Finish rate     : Pre-Trained mean={pre_finish_mean:.4f}  "
              f"Attention mean={att_finish_mean:.4f}  "
              f"=> % decline = {finish_pct_decline:+.2f}%")


if __name__ == "__main__":
    main()