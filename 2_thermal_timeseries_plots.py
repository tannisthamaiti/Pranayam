"""
=============================================================
CODE 2: Thermal Value Time-Series Plotter
=============================================================
Run AFTER Code 1. Reads nose_tracking.csv and produces:
  - nose_thermal_timeseries.png   (all 9 points over time)
  - nose_thermal_heatmap.png      (point × time heatmap)
  - nose_thermal_correlation.png  (correlation matrix)

Requires: nose_tracking.csv (output of Code 1)
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import warnings
warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────
INPUT_CSV = "nose_tracking.csv"       # from Code 1

NOSE_LABELS = {
    "nose_tip":         "Nose Tip",
    "nose_bridge":      "Nose Bridge",
    "left_nostril":     "Left Nostril",
    "right_nostril":    "Right Nostril",
    "nose_left_ala":    "Left Ala",
    "nose_right_ala":   "Right Ala",
    "philtrum":         "Philtrum",
    "nose_base_left":   "Base Left",
    "nose_base_right":  "Base Right",
}

COLORS = [
    "#e74c3c","#e67e22","#f1c40f","#2ecc71","#1abc9c",
    "#3498db","#9b59b6","#e91e63","#795548"
]


def plot_timeseries(df_clean):
    """9-panel time-series: each nose point's thermal intensity vs time."""
    fig, axes = plt.subplots(3, 3, figsize=(18, 12), sharex=True)
    fig.suptitle("Thermal Intensity Over Time — Nose Region Points", 
                 fontsize=16, fontweight="bold", y=1.01)

    t = df_clean["timestamp_s"] / 60   # convert to minutes

    for ax, (key, label), color in zip(axes.flat, NOSE_LABELS.items(), COLORS):
        col = f"{key}_TI"
        if col not in df_clean.columns:
            ax.set_visible(False)
            continue

        vals = df_clean[col]

        # raw + smoothed (rolling 10-sec window)
        smoothed = vals.rolling(10, center=True, min_periods=1).mean()

        ax.plot(t, vals, alpha=0.25, color=color, linewidth=0.8, label="raw")
        ax.plot(t, smoothed, color=color, linewidth=2, label="smoothed")
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_ylabel("Thermal Intensity (0–255)", fontsize=8)
        ax.set_ylim(0, 260)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")

    for ax in axes[-1]:
        ax.set_xlabel("Time (minutes)", fontsize=9)

    plt.tight_layout()
    plt.savefig("nose_thermal_timeseries.png", dpi=150, bbox_inches="tight")
    print("Saved: nose_thermal_timeseries.png")
    plt.close()


def plot_heatmap(df_clean):
    """Heatmap: rows = nose points, columns = time bins."""
    ti_cols   = [f"{k}_TI" for k in NOSE_LABELS]
    available = [c for c in ti_cols if c in df_clean.columns]
    labels    = [NOSE_LABELS[c.replace("_TI", "")] for c in available]

    # Bin into 1-minute windows
    df_clean = df_clean.copy()
    df_clean["minute"] = (df_clean["timestamp_s"] // 60).astype(int)
    heat = df_clean.groupby("minute")[available].mean().T
    heat.index = labels

    fig, ax = plt.subplots(figsize=(20, 5))
    im = ax.imshow(heat.values, aspect="auto", cmap="inferno",
                   norm=Normalize(vmin=heat.values.min(), vmax=heat.values.max()))
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Time (minutes)", fontsize=11)
    ax.set_title("Thermal Intensity Heatmap — Nose Points × Time", 
                 fontsize=13, fontweight="bold")
    ax.set_xticks(range(heat.shape[1]))
    ax.set_xticklabels(heat.columns, rotation=45, ha="right", fontsize=8)
    plt.colorbar(im, ax=ax, label="Thermal Intensity")
    plt.tight_layout()
    plt.savefig("nose_thermal_heatmap.png", dpi=150, bbox_inches="tight")
    print("Saved: nose_thermal_heatmap.png")
    plt.close()


def plot_correlation(df_clean):
    """Correlation matrix between nose landmark thermal intensities."""
    ti_cols   = [f"{k}_TI" for k in NOSE_LABELS]
    available = [c for c in ti_cols if c in df_clean.columns]
    labels    = [NOSE_LABELS[c.replace("_TI", "")] for c in available]

    corr = df_clean[available].corr()
    corr.columns = labels
    corr.index   = labels

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center",
                    fontsize=8, color="black" if abs(corr.values[i,j]) < 0.6 else "white")
    plt.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title("Correlation Between Nose Point Thermal Intensities",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("nose_thermal_correlation.png", dpi=150, bbox_inches="tight")
    print("Saved: nose_thermal_correlation.png")
    plt.close()


def plot_delta_vs_bridge(df_clean):
    """How much each point deviates from the nose bridge (baseline)."""
    if "nose_bridge_TI" not in df_clean.columns:
        return
    fig, ax = plt.subplots(figsize=(14, 5))
    t = df_clean["timestamp_s"] / 60
    bridge = df_clean["nose_bridge_TI"]

    skip = {"nose_bridge"}
    for (key, label), color in zip(NOSE_LABELS.items(), COLORS):
        if key in skip:
            continue
        col = f"{key}_TI"
        if col not in df_clean.columns:
            continue
        delta = df_clean[col] - bridge
        smooth = delta.rolling(10, center=True, min_periods=1).mean()
        ax.plot(t, smooth, label=label, color=color, linewidth=1.5)

    ax.axhline(0, color="black", linewidth=1, linestyle="--", label="Bridge baseline")
    ax.set_xlabel("Time (minutes)", fontsize=11)
    ax.set_ylabel("Thermal Δ from Nose Bridge", fontsize=11)
    ax.set_title("Relative Thermal Variation — Each Nose Point vs Bridge", fontsize=13)
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("nose_thermal_delta.png", dpi=150, bbox_inches="tight")
    print("Saved: nose_thermal_delta.png")
    plt.close()


def main():
    print(f"Loading {INPUT_CSV} …")
    df = pd.read_csv(INPUT_CSV)
    print(f"  {len(df)} rows, {len(df.columns)} columns")

    # Keep only rows with face detected
    df_clean = df[df["face_detected"] == True].copy().reset_index(drop=True)
    print(f"  {len(df_clean)} rows with face detected")

    if df_clean.empty:
        print("No face-detected rows found. Check your tracking CSV.")
        return

    plot_timeseries(df_clean)
    plot_heatmap(df_clean)
    plot_correlation(df_clean)
    plot_delta_vs_bridge(df_clean)

    # ── Descriptive stats table ────────────────────────────
    ti_cols = [c for c in df_clean.columns if c.endswith("_TI")]
    stats = df_clean[ti_cols].describe().T
    stats.index = [NOSE_LABELS.get(i.replace("_TI",""), i) for i in stats.index]
    print("\n── Thermal Intensity Statistics ──")
    print(stats[["mean","std","min","max"]].round(2).to_string())


if __name__ == "__main__":
    main()
