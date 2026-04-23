"""
Plot nostril_predictions_inference.csv onto thermal frames from check_thermal.mp4.
Saves:
  inference_pred_vis/   -- individual annotated frames
  inference_pred_sheet.jpg -- 4-column contact sheet
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path

# ── CONFIG ─────────────────────────────────────────────────────────────────────
VIDEO_PATH = "check_thermal.mp4"
CSV_PATH   = "nostril_predictions_inference.csv"
OUT_DIR    = Path("inference_pred_vis")
SHEET_OUT  = "inference_pred_sheet.jpg"
COLS       = 4

# Composite layout — must match extract_thermal_frames.py
FACE_X1, FACE_Y1 = 200, 50
FACE_W,  FACE_H  = 250, 200
NOSE_X1, NOSE_Y1 = 260, 145
NOSE_W,  NOSE_H  = 130, 90
ZOOM              = 6

NZ_H             = NOSE_H * ZOOM          # 540
NZ_W             = NOSE_W * ZOOM          # 780
FACE_PANEL_SCALE = NZ_H / FACE_H          # 2.7
FACE_PANEL_W     = int(FACE_W * FACE_PANEL_SCALE)   # 675

COL_LEFT  = (255, 80,  20)
COL_RIGHT = (20,  80, 255)
COL_LINE  = (180, 180, 180)
RADIUS    = 10
FACE_RADIUS = max(3, int(RADIUS / ZOOM * FACE_PANEL_SCALE))

# ── HELPERS ────────────────────────────────────────────────────────────────────

def build_composite(frame):
    face      = frame[FACE_Y1:FACE_Y1+FACE_H, FACE_X1:FACE_X1+FACE_W].copy()
    nose_zoom = cv2.resize(
        frame[NOSE_Y1:NOSE_Y1+NOSE_H, NOSE_X1:NOSE_X1+NOSE_W],
        (NZ_W, NZ_H), interpolation=cv2.INTER_CUBIC
    )
    face_r = cv2.resize(face, (FACE_PANEL_W, NZ_H), interpolation=cv2.INTER_CUBIC)

    nfx1 = int((NOSE_X1 - FACE_X1) * FACE_PANEL_SCALE)
    nfx2 = int((NOSE_X1 - FACE_X1 + NOSE_W) * FACE_PANEL_SCALE)
    nfy1 = int((NOSE_Y1 - FACE_Y1) * FACE_PANEL_SCALE)
    nfy2 = int((NOSE_Y1 - FACE_Y1 + NOSE_H) * FACE_PANEL_SCALE)
    cv2.rectangle(face_r, (nfx1, nfy1), (nfx2, nfy2), (255, 255, 255), 1)

    return np.hstack([face_r, nose_zoom])


def orig_to_face_panel(ox, oy):
    return int((ox - FACE_X1) * FACE_PANEL_SCALE), int((oy - FACE_Y1) * FACE_PANEL_SCALE)


def orig_to_nose_panel(ox, oy):
    return int((ox - NOSE_X1) * ZOOM), int((oy - NOSE_Y1) * ZOOM)


def draw_point(img, x, y, color, r, label=""):
    cv2.circle(img, (x, y), r,     color, -1)
    cv2.circle(img, (x, y), r + 2, (255, 255, 255), 1)
    if label:
        cv2.putText(img, label, (x + r + 3, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def annotate(composite, lox, loy, rox, roy, frame_idx):
    out = composite.copy()

    # Face panel
    lf = orig_to_face_panel(lox, loy)
    rf = orig_to_face_panel(rox, roy)
    draw_point(out, lf[0], lf[1], COL_LEFT,  FACE_RADIUS, "L")
    draw_point(out, rf[0], rf[1], COL_RIGHT, FACE_RADIUS, "R")

    # Nose zoom panel
    ln = orig_to_nose_panel(lox, loy)
    rn = orig_to_nose_panel(rox, roy)
    ln_d = (ln[0] + FACE_PANEL_W, ln[1])
    rn_d = (rn[0] + FACE_PANEL_W, rn[1])
    if 0 <= ln[0] < NZ_W and 0 <= ln[1] < NZ_H:
        draw_point(out, ln_d[0], ln_d[1], COL_LEFT,  RADIUS, "L")
    if 0 <= rn[0] < NZ_W and 0 <= rn[1] < NZ_H:
        draw_point(out, rn_d[0], rn_d[1], COL_RIGHT, RADIUS, "R")
    if (0 <= ln[0] < NZ_W and 0 <= ln[1] < NZ_H and
            0 <= rn[0] < NZ_W and 0 <= rn[1] < NZ_H):
        cv2.line(out, ln_d, rn_d, COL_LINE, 1)

    cv2.line(out, (FACE_PANEL_W, 0), (FACE_PANEL_W, out.shape[0]), (80, 80, 80), 1)
    cv2.putText(out, f"f{frame_idx}", (4, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    return out

# ── MAIN ───────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} predictions")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open {VIDEO_PATH}")

    annotated_imgs = []

    for _, row in df.iterrows():
        # Parse frame index from filename: "frame_00042.jpg" -> 42
        fi = int(Path(row["frame_file"]).stem.split("_")[1])

        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            print(f"  [SKIP] frame {fi} unreadable")
            continue

        composite = build_composite(frame)
        ann = annotate(composite,
                       row["pred_left_orig_x"],  row["pred_left_orig_y"],
                       row["pred_right_orig_x"], row["pred_right_orig_y"],
                       fi)

        cv2.imwrite(str(OUT_DIR / f"pred_frame_{fi:05d}.jpg"), ann)
        annotated_imgs.append(ann)

    cap.release()
    print(f"Saved {len(annotated_imgs)} frames to {OUT_DIR}/")

    # Contact sheet
    thumb_h = 200
    thumbs  = [cv2.resize(img, (int(img.shape[1] * thumb_h / img.shape[0]), thumb_h),
                          interpolation=cv2.INTER_AREA)
               for img in annotated_imgs]

    tw = thumbs[0].shape[1]
    rows_needed = (len(thumbs) + COLS - 1) // COLS
    sheet = np.zeros((thumb_h * rows_needed, tw * COLS, 3), dtype=np.uint8)
    for i, t in enumerate(thumbs):
        r, c = i // COLS, i % COLS
        sheet[r*thumb_h:(r+1)*thumb_h, c*tw:(c+1)*tw] = t

    cv2.imwrite(SHEET_OUT, sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"Contact sheet -> {SHEET_OUT}  ({COLS} cols x {rows_needed} rows)")


if __name__ == "__main__":
    main()
