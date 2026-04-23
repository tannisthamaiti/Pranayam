"""
=============================================================
CODE 4: Posture Analysis (2D + 3D Comparison)
=============================================================
Reads posture_analysis_full.csv  (2D keypoint posture)
      3D_posture_analysis_full.csv (3D posture)

Produces:
  - posture_score_timeline.png
  - posture_angle_vs_kpdist.png
  - posture_2d_vs_3d.png

Columns:
  Minute_Start_Sec, PostureID, Avg_Score, Avg_MeanAngleDiff,
  Avg_MeanKPDist, Worst_Image_Path, Worst_Annotations, Corrective_Suggestion
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

CSV_2D = "NEEMA_Day1-20260408T100839Z-3-001\\NEEMA_Day1\\backup_2025-12-26_12-53-29_NEEMA_DAY1\\posture_analysis_full.csv"
CSV_3D = "NEEMA_Day1-20260408T100839Z-3-001\\NEEMA_Day1\\backup_2025-12-26_12-53-29_NEEMA_DAY1\\3D_posture_analysis_full.csv"


def load_data():
    df2 = pd.read_csv(CSV_2D)
    df3 = pd.read_csv(CSV_3D)
    for df in [df2, df3]:
        df["time_min"] = df["Minute_Start_Sec"] / 60
        df["has_posture"] = df["PostureID"].notna() & (df["PostureID"] != "N/A")
    return df2, df3


def score_color(score):
    if score >= 80:   return "#2ecc71"   # good
    elif score >= 60: return "#f39c12"   # moderate
    else:             return "#e74c3c"   # poor


def plot_score_timeline(df2, df3):
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    fig.suptitle("Posture Score Timeline", fontsize=14, fontweight="bold")

    for ax, df, title, color in [
        (axes[0], df2, "2D Posture Score", "#3498db"),
        (axes[1], df3, "3D Posture Score", "#9b59b6"),
    ]:
        bars = ax.bar(df["time_min"], df["Avg_Score"],
                      width=0.8, color=[score_color(s) for s in df["Avg_Score"]])
        ax.axhline(80, color="#2ecc71", linewidth=1.5, linestyle="--", label="Good threshold (80)")
        ax.axhline(60, color="#e74c3c", linewidth=1.5, linestyle="--", label="Poor threshold (60)")
        ax.set_ylabel("Avg Score (0–100)", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 110)
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

        # Annotate suggestions for poor posture
        for _, row in df.iterrows():
            if row["Avg_Score"] < 70 and row["has_posture"]:
                ax.text(row["time_min"], row["Avg_Score"] + 2,
                        row.get("Corrective_Suggestion","")[:20],
                        fontsize=6, ha="center", color="#c0392b", rotation=45)

    axes[-1].set_xlabel("Time (minutes)", fontsize=10)

    # legend patches
    patches = [
        mpatches.Patch(color="#2ecc71", label="Good (≥80)"),
        mpatches.Patch(color="#f39c12", label="Moderate (60–80)"),
        mpatches.Patch(color="#e74c3c", label="Poor (<60)"),
    ]
    axes[0].legend(handles=patches + axes[0].get_lines(), fontsize=8, loc="upper right")

    plt.tight_layout()
    plt.savefig("posture_score_timeline.png", dpi=150, bbox_inches="tight")
    print("Saved: posture_score_timeline.png")
    plt.close()


def plot_angle_vs_kpdist(df2, df3):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Angle Error vs Keypoint Distance Error", fontsize=13, fontweight="bold")

    for ax, df, title, color in [
        (axes[0], df2, "2D Analysis", "#3498db"),
        (axes[1], df3, "3D Analysis", "#9b59b6"),
    ]:
        sc = ax.scatter(df["Avg_MeanAngleDiff"], df["Avg_MeanKPDist"],
                        c=df["Avg_Score"], cmap="RdYlGn",
                        s=80, edgecolors="grey", linewidth=0.5, vmin=0, vmax=100)
        plt.colorbar(sc, ax=ax, label="Avg Score")

        # Label time
        for _, row in df.iterrows():
            ax.annotate(f"{row['time_min']:.0f}m",
                        (row["Avg_MeanAngleDiff"], row["Avg_MeanKPDist"]),
                        fontsize=6, ha="center", va="bottom")

        ax.set_xlabel("Mean Angle Difference (°)", fontsize=10)
        ax.set_ylabel("Mean Keypoint Distance", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("posture_angle_vs_kpdist.png", dpi=150, bbox_inches="tight")
    print("Saved: posture_angle_vs_kpdist.png")
    plt.close()


def plot_2d_vs_3d(df2, df3):
    merged = pd.merge(
        df2[["time_min","Avg_Score","Avg_MeanAngleDiff","Avg_MeanKPDist"]],
        df3[["time_min","Avg_Score","Avg_MeanAngleDiff","Avg_MeanKPDist"]],
        on="time_min", suffixes=("_2D","_3D")
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("2D vs 3D Posture Comparison", fontsize=13, fontweight="bold")

    pairs = [
        ("Avg_Score_2D",        "Avg_Score_3D",        "Score"),
        ("Avg_MeanAngleDiff_2D","Avg_MeanAngleDiff_3D","Mean Angle Diff (°)"),
        ("Avg_MeanKPDist_2D",   "Avg_MeanKPDist_3D",  "Mean KP Distance"),
    ]
    for ax, (col2d, col3d, ylabel) in zip(axes, pairs):
        ax.plot(merged["time_min"], merged[col2d], "o-", color="#3498db",
                linewidth=1.5, markersize=4, label="2D")
        ax.plot(merged["time_min"], merged[col3d], "s--", color="#9b59b6",
                linewidth=1.5, markersize=4, label="3D")
        ax.set_xlabel("Time (min)"); ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(ylabel, fontsize=10, fontweight="bold")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("posture_2d_vs_3d.png", dpi=150, bbox_inches="tight")
    print("Saved: posture_2d_vs_3d.png")
    plt.close()


def print_suggestions(df2, df3):
    print("\n── Corrective Suggestions (2D) ──")
    for _, row in df2.iterrows():
        s = row.get("Corrective_Suggestion","")
        if s and s != "Good form":
            print(f"  t={row['time_min']:.0f}m  [{row['PostureID']}]  → {s}")

    print("\n── Corrective Suggestions (3D) ──")
    for _, row in df3.iterrows():
        s = row.get("Corrective_Suggestion","")
        if s and s != "Good form":
            print(f"  t={row['time_min']:.0f}m  [{row['PostureID']}]  → {s}")


def main():
    df2, df3 = load_data()
    print(f"2D posture: {len(df2)} minutes  |  avg score={df2['Avg_Score'].mean():.1f}")
    print(f"3D posture: {len(df3)} minutes  |  avg score={df3['Avg_Score'].mean():.1f}")
    plot_score_timeline(df2, df3)
    plot_angle_vs_kpdist(df2, df3)
    plot_2d_vs_3d(df2, df3)
    print_suggestions(df2, df3)


if __name__ == "__main__":
    main()
