"""
=============================================================
CODE 6: Multi-Modal Fusion Dashboard
=============================================================
Combines ALL data sources onto a single aligned timeline:
  - Thermal nose temperatures (from Code 1 output)
  - Radar Heart Rate + HRV
  - Radar Breathing Rate
  - Posture Score
  - CD6 raw signal (resampled)

Produces:
  - multimodal_dashboard.png   (full aligned dashboard)
  - multimodal_correlation.png (cross-modal correlation)

Run AFTER Code 1 has generated nose_tracking.csv
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings("ignore")

# ── Input files ───────────────────────────────────────────
THERMAL_CSV = "nose_tracking.csv"          # from Code 1
HEART_CSV   = "NEEMA_Day1-20260408T100839Z-3-001\\NEEMA_Day1\\backup_2025-12-26_12-53-29_NEEMA_DAY1\\Radar_Heart_Output_1.csv"
BREATH_CSV  = "NEEMA_Day1-20260408T100839Z-3-001\\NEEMA_Day1\\backup_2025-12-26_12-53-29_NEEMA_DAY1\\Radar_Breath_Output_1.csv"
POSTURE_2D  = "NEEMA_Day1-20260408T100839Z-3-001\\NEEMA_Day1\\backup_2025-12-26_12-53-29_NEEMA_DAY1\\posture_analysis_full.csv"
CD6_CSV     = "NEEMA_Day1-20260408T100839Z-3-001\\NEEMA_Day1\\backup_2025-12-26_12-53-29_NEEMA_DAY1\\cd6_write.csv"


def load_and_align():
    """Load all CSVs and resample to a common 15-second grid."""

    # ── time grid: 0–1417 seconds, step 15s ──────────────
    t_grid = np.arange(15, 1420, 15)

    def interp_series(t_src, y_src, t_tgt):
        """Linear interpolation, extrapolation clipped."""
        mask = np.isfinite(y_src)
        if mask.sum() < 2:
            return np.full(len(t_tgt), np.nan)
        f = interp1d(t_src[mask], y_src[mask], bounds_error=False, fill_value=np.nan)
        return f(t_tgt)

    fused = pd.DataFrame({"time_s": t_grid, "time_min": t_grid / 60})

    # ── Thermal ──────────────────────────────────────────
    try:
        th = pd.read_csv(THERMAL_CSV)
        th = th[th["face_detected"] == True].copy()
        for col in ["nose_tip_TI", "nose_bridge_TI", "left_nostril_TI",
                    "right_nostril_TI", "nose_left_ala_TI", "nose_right_ala_TI"]:
            if col in th.columns:
                fused[col] = interp_series(th["timestamp_s"].values,
                                           th[col].values, t_grid)
        print("✓ Thermal loaded")
    except FileNotFoundError:
        print("⚠ nose_tracking.csv not found — run Code 1 first for thermal data")

    # ── Heart ─────────────────────────────────────────────
    hr = pd.read_csv(HEART_CSV)
    fused["HR"]      = interp_series(hr["Time_HR"].values, hr["HR"].values, t_grid)
    fused["RMSSD"]   = interp_series(hr["Time_HR"].values, hr["RMSSD"].values, t_grid)
    fused["pNN50"]   = interp_series(hr["Time_HR"].values, hr["pNN50"].values, t_grid)
    fused["LF_HF"]   = interp_series(hr["Time_HR"].values, hr["LF_by_HF"].values, t_grid)
    print("✓ Heart loaded")

    # ── Breath ────────────────────────────────────────────
    br = pd.read_csv(BREATH_CSV)
    fused["BR"]      = interp_series(br["Time_BR"].values, br["BR"].values, t_grid)
    fused["RMSSD_B"] = interp_series(br["Time_BR"].values, br["RMSSD_B"].values, t_grid)
    print("✓ Breath loaded")

    # ── Posture ───────────────────────────────────────────
    p2 = pd.read_csv(POSTURE_2D)
    fused["Posture_Score"] = interp_series(
        p2["Minute_Start_Sec"].values, p2["Avg_Score"].values, t_grid
    )
    print("✓ Posture loaded")

    # ── CD6 (downsample to grid) ──────────────────────────
    cd6 = pd.read_csv(CD6_CSV).sort_values("time")
    # Rolling mean at 10s windows to align with grid
    cd6["bin"] = (cd6["time"] // 15) * 15 + 15
    cd6_agg = cd6.groupby("bin")["data"].mean().reset_index()
    cd6_agg.columns = ["time_bin", "cd6_mean"]
    fused["CD6"] = interp_series(cd6_agg["time_bin"].values,
                                 cd6_agg["cd6_mean"].values, t_grid)
    print("✓ CD6 loaded")

    return fused


def plot_dashboard(fused):
    thermal_cols = [c for c in fused.columns if c.endswith("_TI")]
    has_thermal  = len(thermal_cols) > 0
    n_panels     = 5 + (1 if has_thermal else 0)

    fig = plt.figure(figsize=(20, 4 * n_panels))
    gs  = gridspec.GridSpec(n_panels, 1, hspace=0.5)
    fig.suptitle("Multi-Modal Physiological Dashboard", fontsize=16, fontweight="bold")

    t = fused["time_min"]
    panel_idx = 0

    # Panel: Thermal nose region
    if has_thermal:
        ax = fig.add_subplot(gs[panel_idx]); panel_idx += 1
        colors = ["#e74c3c","#e67e22","#f1c40f","#2ecc71","#3498db","#9b59b6"]
        for col, color in zip(thermal_cols, colors):
            label = col.replace("_TI","").replace("_"," ").title()
            ax.plot(t, fused[col], alpha=0.6, linewidth=1, color=color, label=label)
        ax.set_ylabel("Thermal Intensity", fontsize=9)
        ax.set_title("Thermal — Nose Region Points", fontsize=10, fontweight="bold")
        ax.legend(ncol=3, fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

    # Panel: Heart Rate
    ax = fig.add_subplot(gs[panel_idx]); panel_idx += 1
    ax.plot(t, fused["HR"], color="#e74c3c", linewidth=1.5, label="Heart Rate (bpm)")
    ax.set_ylabel("HR (bpm)", fontsize=9); ax.legend(fontsize=8)
    ax.set_title("Heart Rate", fontsize=10, fontweight="bold"); ax.grid(True, alpha=0.3)

    # Panel: HRV
    ax = fig.add_subplot(gs[panel_idx]); panel_idx += 1
    ax2 = ax.twinx()
    ax.plot(t, fused["RMSSD"], color="#9b59b6", linewidth=1.5, label="RMSSD")
    ax2.plot(t, fused["pNN50"], color="#1abc9c", linewidth=1.5, linestyle="--", label="pNN50")
    ax.set_ylabel("RMSSD (ms)", fontsize=9); ax2.set_ylabel("pNN50 (%)", fontsize=9)
    ax.set_title("HRV Metrics", fontsize=10, fontweight="bold")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel: Breathing Rate
    ax = fig.add_subplot(gs[panel_idx]); panel_idx += 1
    ax.plot(t, fused["BR"], color="#3498db", linewidth=1.5, label="Breathing Rate (br/min)")
    ax.set_ylabel("BR (br/min)", fontsize=9); ax.legend(fontsize=8)
    ax.set_title("Breathing Rate", fontsize=10, fontweight="bold"); ax.grid(True, alpha=0.3)

    # Panel: Posture Score
    ax = fig.add_subplot(gs[panel_idx]); panel_idx += 1
    ax.fill_between(t, fused["Posture_Score"], alpha=0.3, color="#27ae60")
    ax.plot(t, fused["Posture_Score"], color="#27ae60", linewidth=1.5, label="Posture Score")
    ax.axhline(80, color="green",  linewidth=1, linestyle="--", alpha=0.7)
    ax.axhline(60, color="orange", linewidth=1, linestyle="--", alpha=0.7)
    ax.set_ylabel("Score (0–100)", fontsize=9); ax.legend(fontsize=8)
    ax.set_title("2D Posture Score", fontsize=10, fontweight="bold"); ax.grid(True, alpha=0.3)

    # Panel: CD6
    ax = fig.add_subplot(gs[panel_idx]); panel_idx += 1
    ax.plot(t, fused["CD6"], color="#2c3e50", linewidth=0.8, alpha=0.8, label="CD6 Signal")
    ax.set_ylabel("CD6 Value", fontsize=9); ax.legend(fontsize=8)
    ax.set_title("CD6 Raw Signal", fontsize=10, fontweight="bold"); ax.grid(True, alpha=0.3)
    ax.set_xlabel("Time (minutes)", fontsize=11)

    plt.savefig("multimodal_dashboard.png", dpi=150, bbox_inches="tight")
    print("Saved: multimodal_dashboard.png")
    plt.close()


def plot_cross_modal_correlation(fused):
    cols_of_interest = {
        "nose_tip_TI":    "Nose Tip Thermal",
        "left_nostril_TI":"Left Nostril Thermal",
        "HR":             "Heart Rate",
        "RMSSD":          "RMSSD (HRV)",
        "pNN50":          "pNN50",
        "LF_HF":          "LF/HF",
        "BR":             "Breathing Rate",
        "Posture_Score":  "Posture Score",
        "CD6":            "CD6 Signal",
    }
    available = {k: v for k, v in cols_of_interest.items() if k in fused.columns}
    sub = fused[list(available.keys())].copy()
    sub.columns = list(available.values())
    corr = sub.corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1)
    labels = list(available.values())
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="black" if abs(v) < 0.6 else "white")
    plt.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title("Cross-Modal Correlation Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("multimodal_correlation.png", dpi=150, bbox_inches="tight")
    print("Saved: multimodal_correlation.png")
    plt.close()


def main():
    fused = load_and_align()
    print(f"\nFused dataset: {len(fused)} rows  |  {fused.columns.tolist()}")
    fused.to_csv("fused_multimodal.csv", index=False)
    print("Saved: fused_multimodal.csv")
    plot_dashboard(fused)
    plot_cross_modal_correlation(fused)


if __name__ == "__main__":
    main()
