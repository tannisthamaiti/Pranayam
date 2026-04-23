"""
=============================================================
CODE 3: Radar Heart Rate & Breathing Analysis
=============================================================
Reads Radar_Heart_Output_1.csv and Radar_Breath_Output_1.csv
Produces:
  - radar_vitals_overview.png
  - radar_hrv_metrics.png
  - radar_breathing_metrics.png
  - radar_freq_bands.png

Columns in Heart CSV:
  Time_HR, HR, Mean_RR, STD_RR, Mean_HR, STD_HR,
  Min_HR, Max_HR, RMSSD, NN50, pNN50,
  Power_VLF, Power_LF, Power_HF, Power_Total, LF_by_HF,
  Peak_VLF, Peak_LF, Peak_HF, Fraction_LF, Fraction_HF

Columns in Breath CSV:
  Time_BR, BR, Mean_RR_B, STD_RR_B, Mean_HR_B, STD_HR_B,
  Min_HR_B, Max_HR_B, RMSSD_B, NN50_B, pNN50_B,
  Power_VLF_B ... (HRV from breathing signal)
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

HEART_CSV  = "NEEMA_Day1-20260408T100839Z-3-001\\NEEMA_Day1\\backup_2025-12-26_12-53-29_NEEMA_DAY1\\Radar_Heart_Output_1.csv"
BREATH_CSV = "NEEMA_Day1-20260408T100839Z-3-001\\NEEMA_Day1\\backup_2025-12-26_12-53-29_NEEMA_DAY1\\Radar_Breath_Output_1.csv"


def load_data():
    hr = pd.read_csv(HEART_CSV)
    br = pd.read_csv(BREATH_CSV)
    hr["time_min"] = hr["Time_HR"] / 60
    br["time_min"] = br["Time_BR"] / 60
    return hr, br


def plot_vitals_overview(hr, br):
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=False)
    fig.suptitle("Radar Vitals Overview", fontsize=15, fontweight="bold")

    # Panel 1: Heart Rate
    ax = axes[0]
    ax.plot(hr["time_min"], hr["HR"], color="#e74c3c", linewidth=1.5, label="Instantaneous HR")
    ax.plot(hr["time_min"], hr["Mean_HR"], color="#c0392b", linewidth=2.5,
            linestyle="--", label="Mean HR (window)")
    ax.fill_between(hr["time_min"], hr["Min_HR"], hr["Max_HR"],
                    alpha=0.15, color="#e74c3c", label="Min–Max range")
    ax.set_ylabel("Heart Rate (bpm)", fontsize=10)
    ax.set_title("Heart Rate", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_xlabel("Time (min)")

    # Panel 2: Breathing Rate
    ax = axes[1]
    ax.plot(br["time_min"], br["BR"], color="#3498db", linewidth=2, label="Breathing Rate")
    ax.set_ylabel("Breaths / min", fontsize=10)
    ax.set_title("Breathing Rate", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_xlabel("Time (min)")

    # Panel 3: HR std (stress proxy)
    ax = axes[2]
    ax.plot(hr["time_min"], hr["STD_HR"], color="#9b59b6", linewidth=1.5, label="HR Std Dev")
    ax.plot(br["time_min"], br["STD_HR_B"], color="#1abc9c", linewidth=1.5,
            linestyle="--", label="HR Std (breath channel)")
    ax.set_ylabel("HR Std Dev (bpm)", fontsize=10)
    ax.set_title("Heart Rate Variability Proxy (Std Dev)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_xlabel("Time (min)")

    plt.tight_layout()
    plt.savefig("radar_vitals_overview.png", dpi=150, bbox_inches="tight")
    print("Saved: radar_vitals_overview.png")
    plt.close()


def plot_hrv_metrics(hr):
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.suptitle("HRV Metrics (from Radar Heart Channel)", fontsize=14, fontweight="bold")

    metrics = [
        ("RMSSD",    "RMSSD (ms)",           "#e74c3c"),
        ("pNN50",    "pNN50 (%)",             "#e67e22"),
        ("NN50",     "NN50 count",            "#f1c40f"),
        ("Mean_RR",  "Mean RR Interval (ms)", "#2ecc71"),
        ("STD_RR",   "RR Std Dev (ms)",       "#3498db"),
        ("LF_by_HF", "LF/HF Ratio",          "#9b59b6"),
    ]

    for ax, (col, label, color) in zip(axes.flat, metrics):
        if col not in hr.columns:
            ax.set_visible(False); continue
        ax.plot(hr["time_min"], hr[col], color=color, linewidth=1.5)
        smoothed = hr[col].rolling(5, center=True, min_periods=1).mean()
        ax.plot(hr["time_min"], smoothed, color=color, linewidth=2.5, alpha=0.7)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xlabel("Time (min)", fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("radar_hrv_metrics.png", dpi=150, bbox_inches="tight")
    print("Saved: radar_hrv_metrics.png")
    plt.close()


def plot_breathing_metrics(br):
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.suptitle("HRV Metrics from Breathing Channel (Radar)", fontsize=14, fontweight="bold")

    metrics = [
        ("RMSSD_B",   "RMSSD (ms)",            "#3498db"),
        ("pNN50_B",   "pNN50 (%)",              "#1abc9c"),
        ("STD_RR_B",  "RR Std Dev (ms)",        "#2ecc71"),
        ("Mean_RR_B", "Mean RR (ms)",           "#f39c12"),
        ("LF_by_HF_B","LF/HF Ratio",           "#e74c3c"),
        ("BR",        "Breathing Rate (br/min)","#9b59b6"),
    ]

    for ax, (col, label, color) in zip(axes.flat, metrics):
        if col not in br.columns:
            ax.set_visible(False); continue
        ax.plot(br["time_min"], br[col], color=color, linewidth=1.5)
        smoothed = br[col].rolling(5, center=True, min_periods=1).mean()
        ax.plot(br["time_min"], smoothed, color=color, linewidth=2.5, alpha=0.7)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xlabel("Time (min)", fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("radar_breathing_metrics.png", dpi=150, bbox_inches="tight")
    print("Saved: radar_breathing_metrics.png")
    plt.close()


def plot_frequency_bands(hr, br):
    """VLF / LF / HF power bands over time."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle("HRV Frequency Band Power Over Time", fontsize=14, fontweight="bold")

    for ax, df, title, suffix in [
        (axes[0], hr, "Heart Channel", ""),
        (axes[1], br, "Breath Channel", "_B"),
    ]:
        vlf = f"Power_VLF{suffix}"; lf = f"Power_LF{suffix}"; hf = f"Power_HF{suffix}"
        if not all(c in df.columns for c in [vlf, lf, hf]):
            ax.set_visible(False); continue
        t = df["time_min"]
        ax.stackplot(t,
                     df[vlf], df[lf], df[hf],
                     labels=["VLF (<0.04 Hz)", "LF (0.04–0.15 Hz)", "HF (0.15–0.4 Hz)"],
                     colors=["#3498db", "#e74c3c", "#2ecc71"], alpha=0.8)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Time (min)"); ax.set_ylabel("Power (ms²)")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("radar_freq_bands.png", dpi=150, bbox_inches="tight")
    print("Saved: radar_freq_bands.png")
    plt.close()


def print_summary(hr, br):
    print("\n── Heart Rate Summary ──")
    for col in ["HR","Mean_HR","RMSSD","pNN50","LF_by_HF"]:
        if col in hr.columns:
            print(f"  {col:<15s} mean={hr[col].mean():.2f}  std={hr[col].std():.2f}  "
                  f"min={hr[col].min():.2f}  max={hr[col].max():.2f}")

    print("\n── Breathing Rate Summary ──")
    for col in ["BR","RMSSD_B","pNN50_B","LF_by_HF_B"]:
        if col in br.columns:
            print(f"  {col:<15s} mean={br[col].mean():.2f}  std={br[col].std():.2f}  "
                  f"min={br[col].min():.2f}  max={br[col].max():.2f}")


def main():
    hr, br = load_data()
    print(f"Heart data : {len(hr)} rows  |  time {hr['Time_HR'].min():.0f}–{hr['Time_HR'].max():.0f} s")
    print(f"Breath data: {len(br)} rows  |  time {br['Time_BR'].min():.0f}–{br['Time_BR'].max():.0f} s")

    plot_vitals_overview(hr, br)
    plot_hrv_metrics(hr)
    plot_breathing_metrics(br)
    plot_frequency_bands(hr, br)
    print_summary(hr, br)


if __name__ == "__main__":
    main()
