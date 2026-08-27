from __future__ import annotations

from pathlib import Path
import pandas as pd

# ============================================================
# SET 1
# Total unsafe counts across all 4 env files combined.
# Compare SFT and Attention against Pre-Trained.
#
# SET 2
# For each environment separately:
#   % decrease = ((Pre-Trained - New) / Pre-Trained) * 100
# Then average those 4 environment-wise % decreases.
# ============================================================

# -----------------------------
# Environment keys
# -----------------------------
ENV_KEYS = ["8", "16", "24", "32"]

# -----------------------------
# All 24 files
# -----------------------------
GCBF_FILES = {
    "Pre-Trained": {
        "8": "episode_metrics_OG_8.csv",
        "16": "episode_metrics_OG_16.csv",
        "24": "episode_metrics_OG_24.csv",
        "32": "episode_metrics_OG_32.csv",
    },
    "SFT": {
        "8": "episode_metrics_SFT_8.csv",
        "16": "episode_metrics_SFT_16.csv",
        "24": "episode_metrics_SFT_24.csv",
        "32": "episode_metrics_SFT_32.csv",
    },
    "Attention": {
        "8": "episode_metrics_Attention_8.csv",
        "16": "episode_metrics_Attention_16.csv",
        "24": "episode_metrics_Attention_24.csv",
        "32": "episode_metrics_Attention_32.csv",
    },
}

MAPPO_FILES = {
    "Pre-Trained": {
        "8": "episode_metrics_MAPPO_OG_8.csv",
        "16": "episode_metrics_MAPPO_OG_16.csv",
        "24": "episode_metrics_MAPPO_OG_24.csv",
        "32": "episode_metrics_MAPPO_OG_32.csv",
    },
    "SFT": {
        "8": "episode_metrics_MAPPO_SFT_8.csv",
        "16": "episode_metrics_MAPPO_SFT_16.csv",
        "24": "episode_metrics_MAPPO_SFT_24.csv",
        "32": "episode_metrics_MAPPO_SFT_32.csv",
    },
    "Attention": {
        "8": "episode_metrics_MAPPO_Attention_8.csv",
        "16": "episode_metrics_MAPPO_Attention_16.csv",
        "24": "episode_metrics_MAPPO_Attention_24.csv",
        "32": "episode_metrics_MAPPO_Attention_32.csv",
    },
}


def load_unsafe_sum(csv_path: str | Path, column_name: str = "labeled_unsafe_count") -> float:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    normalized_cols = {c.strip().lower(): c for c in df.columns}
    key = column_name.strip().lower()
    if key not in normalized_cols:
        raise ValueError(
            f"Column '{column_name}' not found in {csv_path}. "
            f"Available columns: {list(df.columns)}"
        )

    actual_col = normalized_cols[key]
    values = pd.to_numeric(df[actual_col], errors="coerce").dropna()

    if len(values) != 32:
        print(f"Warning: {csv_path.name} has {len(values)} rows in '{actual_col}', expected 32.")

    return float(values.sum())


def framework_env_totals(file_map: dict[str, dict[str, str]]) -> dict[str, dict[str, float]]:
    env_totals = {}
    for variant in ["Pre-Trained", "SFT", "Attention"]:
        env_totals[variant] = {}
        for env_key in ENV_KEYS:
            env_totals[variant][env_key] = load_unsafe_sum(file_map[variant][env_key], "labeled_unsafe_count")
    return env_totals


def framework_totals_from_env_totals(env_totals: dict[str, dict[str, float]]) -> dict[str, float]:
    totals = {}
    for variant in ["Pre-Trained", "SFT", "Attention"]:
        totals[variant] = sum(env_totals[variant][env] for env in ENV_KEYS)
    return totals


def percent_decrease(pretrained_total: float, new_total: float) -> float:
    if pretrained_total == 0:
        raise ZeroDivisionError("Pre-Trained total is zero; percent decrease is undefined.")
    return ((pretrained_total - new_total) / pretrained_total) * 100.0


def compute_set2_average_envwise_decrease(env_totals: dict[str, dict[str, float]], variant: str) -> tuple[float, dict[str, float]]:
    envwise_decreases = {}

    for env in ENV_KEYS:
        pre = env_totals["Pre-Trained"][env]
        new = env_totals[variant][env]

        if pre == 0:
            raise ZeroDivisionError(
                f"Pre-Trained total unsafe count is zero for env {env}; "
                f"env-wise percent decrease is undefined."
            )

        envwise_decreases[env] = ((pre - new) / pre) * 100.0

    avg_decrease = sum(envwise_decreases[env] for env in ENV_KEYS) / len(ENV_KEYS)
    return avg_decrease, envwise_decreases


def print_set1_report(name: str, totals: dict[str, float]) -> None:
    pre = totals["Pre-Trained"]
    sft = totals["SFT"]
    att = totals["Attention"]

    sft_drop = percent_decrease(pre, sft)
    att_drop = percent_decrease(pre, att)

    print(f"\n==================== {name} : SET 1 ====================")
    print("Using one grand total over all 4 environments")
    print(f"Pre-Trained total unsafe count : {pre:.0f}")
    print(f"SFT total unsafe count         : {sft:.0f}")
    print(f"Attention total unsafe count   : {att:.0f}")
    print(f"SFT % decrease over Pre-Trained       : {sft_drop:.2f}%")
    print(f"Attention % decrease over Pre-Trained : {att_drop:.2f}%")


def print_set2_report(name: str, env_totals: dict[str, dict[str, float]]) -> None:
    sft_avg, sft_envwise = compute_set2_average_envwise_decrease(env_totals, "SFT")
    att_avg, att_envwise = compute_set2_average_envwise_decrease(env_totals, "Attention")

    print(f"\n==================== {name} : SET 2 ====================")
    print("Environment-wise % decrease first, then average across 8/16/24/32")

    print("\nSFT env-wise % decreases:")
    for env in ENV_KEYS:
        print(f"  Env {env}: {sft_envwise[env]:.2f}%")
    print(f"SFT average of env-wise % decreases: {sft_avg:.2f}%")

    print("\nAttention env-wise % decreases:")
    for env in ENV_KEYS:
        print(f"  Env {env}: {att_envwise[env]:.2f}%")
    print(f"Attention average of env-wise % decreases: {att_avg:.2f}%")


def main() -> None:
    gcbf_env_totals = framework_env_totals(GCBF_FILES)
    mappo_env_totals = framework_env_totals(MAPPO_FILES)

    gcbf_totals = framework_totals_from_env_totals(gcbf_env_totals)
    mappo_totals = framework_totals_from_env_totals(mappo_env_totals)

    print_set1_report("GCBF+", gcbf_totals)
    print_set2_report("GCBF+", gcbf_env_totals)

    print_set1_report("MAPPO / InforMARL", mappo_totals)
    print_set2_report("MAPPO / InforMARL", mappo_env_totals)


if __name__ == "__main__":
    main()