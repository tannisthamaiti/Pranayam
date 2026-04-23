"""
Validation tool — compare thermal blob detections against manual labels.

Workflow:
  1. Loads nostril_detections_thermal.csv (from 9_thermal_blob_detector.py --preview)
  2. Presents 10 sampled thermal frames one at a time
  3. You click the left nostril, then right nostril, on each frame
  4. Saves your labels and prints error vs. automatic detections
  5. Also compares with MediaPipe RGB coords if available

Controls (per frame):
  Left click  : place next point (left nostril first, then right)
  U           : undo last point
  R           : redo this frame (clear both points)
  Space/Enter : accept and go to next frame
  Q           : quit early (saves whatever was labelled so far)

Outputs:
  manual_nostril_labels.csv   — your clicked coords
  validation_report.png       — error scatter + summary
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import sys, json

# ── CONFIG ─────────────────────────────────────────────────────────────────────
VIDEO_PATH       = "check_thermal.mp4"
DETECTIONS_CSV   = "nostril_detections_thermal.csv"
LABELS_OUT       = "manual_nostril_labels.csv"
REPORT_OUT       = "validation_report.png"
N_FRAMES         = 10       # number of frames to label
DISPLAY_SCALE    = 3.0      # zoom factor for display (native 640x480 is small)

# ── LOAD DETECTIONS ────────────────────────────────────────────────────────────

def load_detections():
    p = Path(DETECTIONS_CSV)
    if not p.exists():
        print(f"[WARN] {DETECTIONS_CSV} not found — run 9_thermal_blob_detector.py first.")
        return pd.DataFrame()
    df = pd.read_csv(p)
    return df[df["face_detected"] == True].reset_index(drop=True)

# ── FRAME SAMPLING ─────────────────────────────────────────────────────────────

def pick_frames(df, n=N_FRAMES):
    """Pick n evenly spaced frames that have detections."""
    if len(df) == 0:
        return []
    idxs = np.linspace(0, len(df)-1, n, dtype=int)
    return [int(df.iloc[i]["frame_idx"]) for i in idxs]

# ── VIDEO READER ───────────────────────────────────────────────────────────────

def read_frame(cap, frame_idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    return frame if ret else None

# ── INTERACTIVE LABELER ────────────────────────────────────────────────────────

WIN = "Nostril Validator  |  Click L then R nostril  |  U=undo  R=redo  Space=next  Q=quit"
POINT_NAMES = ["left_nostril", "right_nostril"]
COLORS = [(255, 100, 0), (0, 100, 255)]   # blue-ish=left, red-ish=right

_clicks   = []
_scale    = DISPLAY_SCALE
_frame_hw = (480, 640)


def _img_to_disp(ix, iy):
    return int(ix * _scale), int(iy * _scale)

def _disp_to_img(dx, dy):
    return dx / _scale, dy / _scale


def _draw(base_disp, det_row):
    vis = base_disp.copy()
    dh, dw = vis.shape[:2]

    # Auto-detection dots (yellow)
    if det_row is not None:
        for col_x, col_y, label in [
            ("left_nostril_px",  "left_nostril_py",  "AUTO L"),
            ("right_nostril_px", "right_nostril_py", "AUTO R"),
        ]:
            if col_x in det_row and not np.isnan(det_row[col_x]):
                dx, dy = _img_to_disp(det_row[col_x], det_row[col_y])
                cv2.circle(vis, (dx, dy), 6, (0, 230, 230), 1)
                cv2.putText(vis, label, (dx+7, dy-4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 230, 230), 1)

    # Clicked points
    for i, (ix, iy) in enumerate(_clicks):
        dx, dy = _img_to_disp(ix, iy)
        col = COLORS[i]
        cv2.circle(vis, (dx, dy), 7, col, -1)
        cv2.putText(vis, POINT_NAMES[i], (dx+8, dy-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

    # Instructions
    if len(_clicks) < 2:
        msg = f"Click: {POINT_NAMES[len(_clicks)]}  ({len(_clicks)+1}/2)"
    else:
        msg = "Both placed — Space/Enter to accept"
    cv2.putText(vis, msg, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
    cv2.putText(vis, "U=undo  R=redo  Space=next  Q=quit",
                (8, dh-8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180,180,180), 1)
    return vis


def _mouse_cb(event, dx, dy, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(_clicks) < 2:
            ix, iy = _disp_to_img(dx, dy)
            _clicks.append((ix, iy))


def label_frame(frame_img, frame_idx, det_row):
    global _clicks
    _clicks = []

    dh = int(frame_img.shape[0] * _scale)
    dw = int(frame_img.shape[1] * _scale)
    base_disp = cv2.resize(frame_img, (dw, dh), interpolation=cv2.INTER_LINEAR)

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, dw, dh)
    cv2.setMouseCallback(WIN, _mouse_cb)

    print(f"\nFrame {frame_idx} — click LEFT nostril, then RIGHT nostril")
    if det_row is not None and not np.isnan(det_row.get("left_nostril_px", np.nan)):
        print(f"  Auto: L=({det_row['left_nostril_px']:.0f},{det_row['left_nostril_py']:.0f})  "
              f"R=({det_row['right_nostril_px']:.0f},{det_row['right_nostril_py']:.0f})")

    while True:
        vis = _draw(base_disp, det_row)
        cv2.imshow(WIN, vis)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('u') or key == ord('U'):
            if _clicks:
                _clicks.pop()

        elif key == ord('r') or key == ord('R'):
            _clicks = []

        elif key in (ord(' '), 13):   # space or enter
            if len(_clicks) == 2:
                break
            print("  (place both points first)")

        elif key == ord('q') or key == ord('Q'):
            cv2.destroyAllWindows()
            return None   # signal early quit

    cv2.destroyAllWindows()
    lx, ly = _clicks[0]
    rx, ry = _clicks[1]
    return {"frame_idx": frame_idx,
            "manual_left_x":  round(lx, 1), "manual_left_y":  round(ly, 1),
            "manual_right_x": round(rx, 1), "manual_right_y": round(ry, 1)}

# ── VALIDATION REPORT ──────────────────────────────────────────────────────────

def compute_errors(labels_df, det_df):
    merged = labels_df.merge(det_df[
        ["frame_idx","left_nostril_px","left_nostril_py",
         "right_nostril_px","right_nostril_py","method"]
    ], on="frame_idx", how="left")

    merged["err_left"]  = np.sqrt(
        (merged.manual_left_x  - merged.left_nostril_px)**2 +
        (merged.manual_left_y  - merged.left_nostril_py)**2)
    merged["err_right"] = np.sqrt(
        (merged.manual_right_x - merged.right_nostril_px)**2 +
        (merged.manual_right_y - merged.right_nostril_py)**2)
    merged["err_mean"]  = (merged.err_left + merged.err_right) / 2
    return merged


def save_report(merged):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Error bars
    ax = axes[0]
    x = np.arange(len(merged))
    ax.bar(x - 0.2, merged.err_left,  0.4, label="Left nostril",  color="#3498db")
    ax.bar(x + 0.2, merged.err_right, 0.4, label="Right nostril", color="#e74c3c")
    ax.set_xticks(x); ax.set_xticklabels(merged.frame_idx, rotation=45, fontsize=8)
    ax.set_ylabel("Pixel error (original frame)"); ax.set_xlabel("Frame index")
    ax.set_title("Per-frame pixel error vs. manual labels")
    ax.legend(); ax.grid(axis="y", alpha=0.3)

    # Scatter: auto vs manual
    ax = axes[1]
    for side, col_m_x, col_m_y, col_a_x, col_a_y, c in [
        ("Left",  "manual_left_x",  "manual_left_y",  "left_nostril_px",  "left_nostril_py",  "#3498db"),
        ("Right", "manual_right_x", "manual_right_y", "right_nostril_px", "right_nostril_py", "#e74c3c"),
    ]:
        ax.scatter(merged[col_m_x], merged[col_m_y], marker="o", color=c,
                   label=f"{side} manual", s=60, alpha=0.8)
        ax.scatter(merged[col_a_x], merged[col_a_y], marker="x", color=c,
                   label=f"{side} auto", s=60, alpha=0.8)
        for _, r in merged.iterrows():
            ax.plot([r[col_m_x], r[col_a_x]], [r[col_m_y], r[col_a_y]],
                    color=c, lw=0.8, alpha=0.5)
    ax.invert_yaxis()
    ax.set_xlabel("x (px)"); ax.set_ylabel("y (px)")
    ax.set_title("Manual (o) vs Auto (x) coords")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # Summary
    ax = axes[2]
    ax.axis("off")
    lines = [
        f"Frames labelled : {len(merged)}",
        f"",
        f"Left nostril",
        f"  mean err = {merged.err_left.mean():.1f} px",
        f"  max  err = {merged.err_left.max():.1f} px",
        f"",
        f"Right nostril",
        f"  mean err = {merged.err_right.mean():.1f} px",
        f"  max  err = {merged.err_right.max():.1f} px",
        f"",
        f"Both nostrils",
        f"  mean err = {merged.err_mean.mean():.1f} px",
        f"",
        f"Methods used:",
    ]
    if "method" in merged.columns:
        for m, cnt in merged["method"].value_counts().items():
            lines.append(f"  {m}: {cnt}")
    verdict = "GOOD — proceed with full run" if merged.err_mean.mean() < 10 \
        else "MARGINAL — consider tuning ROI" if merged.err_mean.mean() < 20 \
        else "POOR — fall back to small-model approach"
    lines += ["", f"Verdict: {verdict}"]
    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
            fontsize=10, va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.suptitle("Thermal Blob Detector — Validation vs Manual Labels",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(REPORT_OUT, dpi=120, bbox_inches="tight")
    print(f"Report saved: {REPORT_OUT}")
    plt.close()


# ── MAIN ───────────────────────────────────────────────────────────────────────

def main():
    det_df = load_detections()

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        sys.exit(f"Cannot open {VIDEO_PATH}")

    frame_idxs = pick_frames(det_df, N_FRAMES)
    print(f"Will label {len(frame_idxs)} frames: {frame_idxs}")

    labels = []
    for fi in frame_idxs:
        frame = read_frame(cap, fi)
        if frame is None:
            print(f"  [SKIP] Frame {fi} unreadable")
            continue

        # Auto-detection row for this frame
        det_row = None
        if len(det_df):
            rows = det_df[det_df["frame_idx"] == fi]
            if len(rows):
                det_row = rows.iloc[0].to_dict()

        result = label_frame(frame, fi, det_row)
        if result is None:
            print("Quit early.")
            break
        labels.append(result)
        print(f"  Saved: L=({result['manual_left_x']:.0f},{result['manual_left_y']:.0f})  "
              f"R=({result['manual_right_x']:.0f},{result['manual_right_y']:.0f})")

    cap.release()

    if not labels:
        print("No labels collected.")
        return

    labels_df = pd.DataFrame(labels)
    labels_df.to_csv(LABELS_OUT, index=False)
    print(f"\nLabels saved: {LABELS_OUT}  ({len(labels_df)} frames)")

    if len(det_df) and len(labels_df):
        merged = compute_errors(labels_df, det_df)
        print("\n--- Pixel errors (auto vs manual) ---")
        print(merged[["frame_idx","err_left","err_right","err_mean","method"]].to_string(index=False))
        print(f"\nMean error: {merged.err_mean.mean():.1f} px")
        save_report(merged)


if __name__ == "__main__":
    main()
