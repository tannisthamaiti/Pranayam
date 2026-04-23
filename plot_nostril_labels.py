"""
Plot nostril_labels.csv landmarks onto thermal face frames.

Reads directly from check_thermal.mp4 (same crop as extract_thermal_frames.py),
draws LEFT (blue) and RIGHT (red) nostrils on both panels, saves:
  nostril_label_vis/   -- individual annotated frames
  nostril_label_sheet.jpg  -- 4-column contact sheet
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path

# ── CONFIG ─────────────────────────────────────────────────────────────────────
VIDEO_PATH  = "check_thermal.mp4"
OUT_DIR     = Path("nostril_label_vis")
SHEET_OUT   = "nostril_label_sheet.jpg"
CSV_PATH    = "nostril_labels.csv"

# Must match extract_thermal_frames.py exactly
FACE_X1, FACE_Y1 = 200, 50          # face crop origin in original 640x480 frame
FACE_W,  FACE_H  = 250, 200         # face crop size
NOSE_X1, NOSE_Y1 = 260, 145         # nose crop origin
NOSE_W,  NOSE_H  = 130, 90          # nose crop size
ZOOM              = 6               # nose zoom factor

# Derived: nose zoom panel size
NZ_H = NOSE_H * ZOOM   # 540
NZ_W = NOSE_W * ZOOM   # 780

# Face panel in composite is resized to same height as nose zoom
FACE_PANEL_SCALE  = NZ_H / FACE_H         # 540 / 200 = 2.7
FACE_PANEL_W      = int(FACE_W * FACE_PANEL_SCALE)   # 675

# Colours (BGR)
COL_LEFT  = (255, 80,  20)   # blue-ish
COL_RIGHT = (20,  80, 255)   # red-ish
COL_LINE  = (180, 180,180)   # grey connector

RADIUS     = 10   # circle radius on nose panel (6x zoom)
FACE_RADIUS = max(3, int(RADIUS / ZOOM * FACE_PANEL_SCALE))  # on face panel

COLS = 4   # contact-sheet columns

# ── COORD HELPERS ──────────────────────────────────────────────────────────────

def orig_to_face_panel(ox, oy):
    """Original 640x480 coords → pixel in face panel of composite image."""
    x = int((ox - FACE_X1) * FACE_PANEL_SCALE)
    y = int((oy - FACE_Y1) * FACE_PANEL_SCALE)
    return x, y


def orig_to_nose_panel(ox, oy):
    """Original 640x480 coords → pixel in nose zoom panel of composite image."""
    x = int((ox - NOSE_X1) * ZOOM)
    y = int((oy - NOSE_Y1) * ZOOM)
    return x, y


def draw_point(img, x, y, color, r, label=""):
    cv2.circle(img, (x, y), r,     color, -1)
    cv2.circle(img, (x, y), r + 2, (255,255,255), 1)
    if label:
        cv2.putText(img, label, (x + r + 3, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def annotate_frame(img, lox, loy, rox, roy, frame_idx):
    out = img.copy()
    h, w = out.shape[:2]

    # ── Face panel ────────────────────────────────────────────────────────────
    lf  = orig_to_face_panel(lox, loy)
    rf  = orig_to_face_panel(rox, roy)

    draw_point(out, lf[0], lf[1], COL_LEFT,  FACE_RADIUS, "L")
    draw_point(out, rf[0], rf[1], COL_RIGHT, FACE_RADIUS, "R")

    # ── Nose zoom panel (offset by FACE_PANEL_W) ─────────────────────────────
    ln = orig_to_nose_panel(lox, loy)
    rn = orig_to_nose_panel(rox, roy)

    ln_disp = (ln[0] + FACE_PANEL_W, ln[1])
    rn_disp = (rn[0] + FACE_PANEL_W, rn[1])

    # Only draw if inside nose panel
    if 0 <= ln[0] < NZ_W and 0 <= ln[1] < NZ_H:
        draw_point(out, ln_disp[0], ln_disp[1], COL_LEFT,  RADIUS, "L")
    if 0 <= rn[0] < NZ_W and 0 <= rn[1] < NZ_H:
        draw_point(out, rn_disp[0], rn_disp[1], COL_RIGHT, RADIUS, "R")
    if (0 <= ln[0] < NZ_W and 0 <= ln[1] < NZ_H and
            0 <= rn[0] < NZ_W and 0 <= rn[1] < NZ_H):
        cv2.line(out, ln_disp, rn_disp, COL_LINE, 1)

    # Separator
    cv2.line(out, (FACE_PANEL_W, 0), (FACE_PANEL_W, h), (80,80,80), 1)

    # Frame label
    cv2.putText(out, f"f{frame_idx}", (4, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,220), 1, cv2.LINE_AA)

    return out


# ── MAIN ───────────────────────────────────────────────────────────────────────

def build_composite(frame):
    """Replicate extract_thermal_frames.py composite layout from a raw video frame."""
    face      = frame[FACE_Y1:FACE_Y1+FACE_H, FACE_X1:FACE_X1+FACE_W].copy()
    nose_zoom = cv2.resize(
        frame[NOSE_Y1:NOSE_Y1+NOSE_H, NOSE_X1:NOSE_X1+NOSE_W],
        (NZ_W, NZ_H), interpolation=cv2.INTER_CUBIC
    )
    face_r = cv2.resize(face, (FACE_PANEL_W, NZ_H), interpolation=cv2.INTER_CUBIC)

    # Nose box outline on face panel
    nfx1 = int((NOSE_X1 - FACE_X1) * FACE_PANEL_SCALE)
    nfx2 = int((NOSE_X1 - FACE_X1 + NOSE_W) * FACE_PANEL_SCALE)
    nfy1 = int((NOSE_Y1 - FACE_Y1) * FACE_PANEL_SCALE)
    nfy2 = int((NOSE_Y1 - FACE_Y1 + NOSE_H) * FACE_PANEL_SCALE)
    cv2.rectangle(face_r, (nfx1, nfy1), (nfx2, nfy2), (255, 255, 255), 1)

    return np.hstack([face_r, nose_zoom])


def main():
    OUT_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    labelled = df[df["skipped"] == False].reset_index(drop=True)
    print(f"Loaded {len(df)} rows  ->  {len(labelled)} labelled frames")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open {VIDEO_PATH}")

    annotated_imgs = []

    for _, row in labelled.iterrows():
        fi = int(row["frame_idx"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            print(f"  [SKIP] frame {fi} unreadable")
            continue

        composite = build_composite(frame)

        ann = annotate_frame(composite,
                             row["left_orig_x"],  row["left_orig_y"],
                             row["right_orig_x"], row["right_orig_y"],
                             fi)

        out_path = OUT_DIR / f"ann_frame_{fi:05d}.jpg"
        cv2.imwrite(str(out_path), ann)
        annotated_imgs.append(ann)

    cap.release()

    print(f"Annotated {len(annotated_imgs)} frames  ->  {OUT_DIR}/")

    if not annotated_imgs:
        print("No frames annotated.")
        return

    # ── Contact sheet ──────────────────────────────────────────────────────────
    thumb_h = 200   # height of each thumbnail in sheet
    thumbs  = []
    for img in annotated_imgs:
        scale = thumb_h / img.shape[0]
        tw    = int(img.shape[1] * scale)
        thumbs.append(cv2.resize(img, (tw, thumb_h), interpolation=cv2.INTER_AREA))

    thumb_w = thumbs[0].shape[1]
    rows_needed = (len(thumbs) + COLS - 1) // COLS
    sheet_w = thumb_w * COLS
    sheet_h = thumb_h * rows_needed
    sheet   = np.zeros((sheet_h, sheet_w, 3), dtype=np.uint8)

    for i, t in enumerate(thumbs):
        r = i // COLS
        c = i  % COLS
        y0, y1 = r * thumb_h, r * thumb_h + thumb_h
        x0, x1 = c * thumb_w, c * thumb_w + thumb_w
        sheet[y0:y1, x0:x1] = t

    cv2.imwrite(SHEET_OUT, sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"Contact sheet  ->  {SHEET_OUT}  ({COLS} cols x {rows_needed} rows)")


if __name__ == "__main__":
    main()
