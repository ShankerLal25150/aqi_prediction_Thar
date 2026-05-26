"""
Exploratory Data Analysis (EDA) - Tharparkar
============================================
Run after backfill to understand your dataset before training.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")

# ── Loader ─────────────────────────────────────────────────────────────────────

def load(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"[EDA] Loaded {len(df)} rows, {df.shape[1]} columns")
    return df

# ── Individual plots ───────────────────────────────────────────────────────────

def plot_aqi_over_time(df: pd.DataFrame, out: str):
    fig, ax = plt.subplots(figsize=(14, 5))
    # Aggregate by mine vs town to avoid a messy spaghetti chart
    sns.lineplot(data=df, x="timestamp", y="aqi", hue="is_mine", 
                 palette={0: "#3b82f6", 1: "#ef4444"}, ax=ax, linewidth=0.8)
    
    ax.axhline(100, color="#ff7e00", linestyle="--", linewidth=1, label="Unhealthy threshold (100)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_ylabel("AQI")
    ax.set_title("AQI Over Time: Mines (Red) vs Residential Towns (Blue)")
    
    # Custom legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[:2] + [handles[-1]], labels=["Town", "Mine", "Threshold"], fontsize=10)
    
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")

def plot_hourly_pattern(df: pd.DataFrame, out: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    # Compare hourly patterns between mines and towns
    sns.barplot(data=df, x="hour", y="aqi", hue="is_mine", 
                palette={0: "#3b82f6", 1: "#ef4444"}, ax=ax, alpha=0.9)
    ax.set_xlabel("Hour of day (UTC)")
    ax.set_ylabel("Mean AQI")
    ax.set_title("Average AQI by Hour of Day (Mines vs Towns)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")

def plot_monthly_boxplot(df: pd.DataFrame, out: str):
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(data=df, x="month", y="aqi", hue="is_mine",
                palette={0: "#3b82f6", 1: "#ef4444"}, ax=ax, fliersize=1)
    ax.set_ylabel("AQI")
    ax.set_title("AQI Distribution by Month (Detecting Dust Seasons)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")

def plot_correlation_heatmap(df: pd.DataFrame, out: str):
    num_cols = [
        "aqi", "pm25", "pm10", "no2", "o3", "co", "so2",
        "temperature", "humidity", "wind_speed", "pressure", 
        "aqi_change_rate", "stagnation_index"
    ]
    existing = [c for c in num_cols if c in df.columns]
    corr = df[existing].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=0.4, ax=ax, annot_kws={"size": 9})
    ax.set_title("Feature Correlation Matrix")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")

def plot_aqi_distribution(df: pd.DataFrame, out: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    sns.histplot(data=df, x="aqi", hue="is_mine", bins=50, 
                 palette={0: "#3b82f6", 1: "#ef4444"}, ax=axes[0], alpha=0.6)
    axes[0].set_title("AQI Histogram")

    # Log transform to check for skew
    df["log_aqi"] = np.log1p(df["aqi"])
    sns.histplot(data=df, x="log_aqi", hue="is_mine", bins=50, 
                 palette={0: "#3b82f6", 1: "#ef4444"}, ax=axes[1], alpha=0.6)
    axes[1].set_title("AQI Log-Transformed")
    
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out}")

# ── Summary stats ──────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame):
    print("\n── AQI category distribution ────────────────────────────")
    bins   = [0, 50, 100, 150, 200, 300, 500]
    labels = ["Good", "Moderate", "Unhealthy_Sens", "Unhealthy", "Very_Unhealthy", "Hazardous"]
    cats   = pd.cut(df["aqi"], bins=bins, labels=labels, right=True)
    counts = cats.value_counts(sort=False)
    
    for cat, cnt in counts.items():
        pct = 100 * cnt / len(df)
        bar = "█" * int(pct / 2)
        print(f"  {cat:15s} {cnt:6d} ({pct:5.1f}%)  {bar}")

    print("\n── Basic stats ──────────────────────────────────────────")
    print(f"  Rows:        {len(df):,}")
    print(f"  Date range:  {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"  AQI mean:    {df['aqi'].mean():.1f}")
    print(f"  AQI max:     {df['aqi'].max():.0f}")

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True, help="Path to backfill .parquet file")
    args = parser.parse_args()

    df = load(args.parquet)
    print_summary(df)

    print("\n[EDA] Generating plots …")
    plot_aqi_over_time(df,       "eda_aqi_over_time.png")
    plot_hourly_pattern(df,      "eda_hourly.png")
    plot_monthly_boxplot(df,     "eda_monthly.png")
    plot_correlation_heatmap(df, "eda_correlation.png")
    plot_aqi_distribution(df,    "eda_distribution.png")

    print("\n[EDA] All plots saved. Open them in VS Code.")

if __name__ == "__main__":
    main()