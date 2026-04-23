"""
Thermal nostril detector — reads check_thermal.mp4 frame by frame.

Pipeline per frame:
  1. Haar cascade face detection  (reinit every HAAR_EVERY frames)
  2. Interpolate / last-known for frames where Haar misses
  3. Within nose ROI (lower-centre of face), search B-channel local minima
     for left + right nostril candidates
  4. Fall back to fixed anatomical ratios when blob search fails
  5. Record coords + multi-channel thermal values

Outputs:
  nostril_detections_thermal.csv   — per-frame nostril coords + signal
  blob_vis_frames/vis_XXXXXX.jpg   — annotated frames (sampled)

Usage:
  python 9_thermal_blob_detector.py               # full run
  python 9_thermal_blob_detector.py --preview 500 # first 500 frames
  python 9_thermal_blob_detector.py --every 5     # every 5th frame
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import argparse, sys

# ── CONFIG ─────────────────────────────────────────────────────────────────────
VIDEO_PATH   = "check_thermal.mp4"
OUT_CSV      = "nostril_detections_thermal.csv"
VIS_DIR      = Path("blob_vis_frames")

# How often to run the (slow) Haar cascade; interpolate between hits
HAAR_EVERY   = 5        # frames between cascade calls

# Fixed anatomical ratios relative to face bbox (calibrated for this subject)
# (rx, ry) = fraction of face w/h from top-left of face box
RATIO_LEFT_NOSTRIL  = (0.36, 0.58)   # slightly left of centre, ~60% down face
RATIO_RIGHT_NOSTRIL = (0.54, 0.58)   # slightly right of centre

# Nose ROI within face bbox (fractions of face w/h)
# Nostrils sit at ~55-68% down the face, 28-72% across
NOSE_ROI_Y1 = 0.52
NOSE_ROI_Y2 = 0.70
NOSE_ROI_X1 = 0.28
NOSE_ROI_X2 = 0.72

# Blob search constraints in *face* pixel units
MIN_NOSTRIL_SEP_F    = 0.06    # min nostril x-separation as fraction of face width
MAX_NOSTRIL_SEP_F    = 0.35
MIN_BLOB_AREA        = 4       # pixels² in nose ROI space
MAX_BLOB_AREA        = 800

# Patch radius for extracting thermal values (in original frame pixels)
PATCH_R = 2

# ── THERMAL SIGNAL ─────────────────────────────────────────────────────────────

def thermal_intensity(patch_bgr):
    """Weighted combo of channels (R/2 + G/4 + B/8), classic proxy for temp."""
    b = patch_bgr[:,:,0].astype(float)
    g = patch_bgr[:,:,1].astype(float)
    r = patch_bgr[:,:,2].astype(float)
    return float((r/2 + g/4 + b/8).mean())

def patch_b(frame, cx, cy, r=PATCH_R):
    """Mean B channel in a small patch; B has dynamic range in warm-face zone."""
    h, w = frame.shape[:2]
    x0 = max(0, cx-r); x1 = min(w, cx+r+1)
    y0 = max(0, cy-r); y1 = min(h, cy+r+1)
    return float(frame[y0:y1, x0:x1, 0].mean())

def patch_ti(frame, cx, cy, r=PATCH_R):
    h, w = frame.shape[:2]
    x0 = max(0, cx-r); x1 = min(w, cx+r+1)
    y0 = max(0, cy-r); y1 = min(h, cy+r+1)
    return thermal_intensity(frame[y0:y1, x0:x1])

# ── FACE TRACKING ──────────────────────────────────────────────────────────────

def make_cascade():
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

def detect_face(frame, cascade, last_face=None):
    """
    Haar cascade on R channel with plausibility filtering.
    Rejects detections that are too large (body heat false-positives in thermal)
    or too far from the last known position.
    """
    h_fr, w_fr = frame.shape[:2]
    r = frame[:,:,2]
    faces = cascade.detectMultiScale(
        r, scaleFactor=1.08, minNeighbors=3, minSize=(55, 55)
    )
    if len(faces) == 0:
        return None

    # Filter: max face size 220px (the real face is ~130-160px in this video)
    # and face centroid must be in upper 60% of frame
    MAX_FACE = 220
    valid = []
    for x, y, w, h in faces:
        if w > MAX_FACE or h > MAX_FACE:
            continue
        cy = y + h // 2
        if cy > h_fr * 0.60:   # face centroid below 60% of frame = body
            continue
        valid.append((x, y, w, h))

    if not valid:
        return None

    # If we have a reference, prefer the closest to last known position
    if last_face is not None:
        lx, ly, lw, lh = last_face
        lcx, lcy = lx + lw // 2, ly + lh // 2
        def dist(f):
            cx, cy = f[0]+f[2]//2, f[1]+f[3]//2
            return (cx-lcx)**2 + (cy-lcy)**2
        valid.sort(key=dist)
        # Only accept if within 100px of last position
        if dist(valid[0]) < 100**2:
            return tuple(valid[0])
        # Otherwise fall back to largest valid face
        return tuple(sorted(valid, key=lambda f: f[2]*f[3], reverse=True)[0])

    return tuple(sorted(valid, key=lambda f: f[2]*f[3], reverse=True)[0])

# ── BLOB-BASED NOSTRIL REFINEMENT ──────────────────────────────────────────────

def find_nostrils_blob(frame, face_box):
    """
    Search the nose ROI (B-channel local minima) for left + right nostrils.
    Returns ((lx,ly),(rx,ry)) in original frame coords, or None.
    """
    fx, fy, fw, fh = face_box
    h_f, w_f = frame.shape[:2]

    # Nose ROI in original frame coords
    roi_x1 = int(max(0,   fx + fw * NOSE_ROI_X1))
    roi_x2 = int(min(w_f, fx + fw * NOSE_ROI_X2))
    roi_y1 = int(max(0,   fy + fh * NOSE_ROI_Y1))
    roi_y2 = int(min(h_f, fy + fh * NOSE_ROI_Y2))
    if roi_x2 <= roi_x1 or roi_y2 <= roi_y1:
        return None

    roi_b = frame[roi_y1:roi_y2, roi_x1:roi_x2, 0].astype(np.float32)

    # Local coolness: large-blur minus original (positive where B is low locally)
    b_blur_big   = cv2.GaussianBlur(roi_b, (21, 21), 0)
    b_blur_small = cv2.GaussianBlur(roi_b, (5,  5),  0)
    local_cool   = np.clip(b_blur_big - b_blur_small, 0, None)

    # Threshold at 30% of max local coolness
    thresh_val = max(3.0, local_cool.max() * 0.30)
    _, mask    = cv2.threshold(local_cool, thresh_val, 255, cv2.THRESH_BINARY)
    mask       = mask.astype(np.uint8)

    # Morphological cleanup
    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for c in cnts:
        area = cv2.contourArea(c)
        if not (MIN_BLOB_AREA <= area <= MAX_BLOB_AREA):
            continue
        M = cv2.moments(c)
        if M["m00"] > 0:
            bx = int(M["m10"] / M["m00"]) + roi_x1
            by = int(M["m01"] / M["m00"]) + roi_y1
            blobs.append((bx, by, area))

    if len(blobs) < 2:
        return None

    # Sort left→right, pick best pair
    blobs.sort(key=lambda b: b[0])
    best_pair  = None
    best_score = -1e9
    min_sep    = fw * MIN_NOSTRIL_SEP_F
    max_sep    = fw * MAX_NOSTRIL_SEP_F

    for i in range(len(blobs)):
        for j in range(i+1, len(blobs)):
            bx1, by1, a1 = blobs[i]
            bx2, by2, a2 = blobs[j]
            sep   = abs(bx2 - bx1)
            ydiff = abs(by2 - by1)
            if not (min_sep <= sep <= max_sep):
                continue
            # Prefer pairs near nose centre and at similar height
            cx_pair = (bx1 + bx2) / 2
            cx_face = fx + fw / 2
            score   = sep - ydiff * 1.5 - abs(cx_pair - cx_face) * 0.3
            if score > best_score:
                best_score = score
                best_pair  = ((bx1, by1), (bx2, by2))

    return best_pair   # None if no valid pair

# ── ANATOMICAL RATIO FALLBACK ──────────────────────────────────────────────────

def nostrils_from_ratios(face_box):
    fx, fy, fw, fh = face_box
    lx = int(fx + RATIO_LEFT_NOSTRIL[0]  * fw)
    ly = int(fy + RATIO_LEFT_NOSTRIL[1]  * fh)
    rx = int(fx + RATIO_RIGHT_NOSTRIL[0] * fw)
    ry = int(fy + RATIO_RIGHT_NOSTRIL[1] * fh)
    return (lx, ly), (rx, ry)

# ── VISUALISATION ──────────────────────────────────────────────────────────────

def draw_vis(frame, face_box, left_pt, right_pt, method, idx):
    vis = frame.copy()
    if face_box:
        x, y, w, h = face_box
        cv2.rectangle(vis, (x,y), (x+w, y+h), (255,255,0), 1)
        # Nose ROI
        rv = lambda frac, sz: int(sz * frac)
        cv2.rectangle(vis,
            (x + rv(NOSE_ROI_X1,w), y + rv(NOSE_ROI_Y1,h)),
            (x + rv(NOSE_ROI_X2,w), y + rv(NOSE_ROI_Y2,h)),
            (200,200,200), 1)
    col = (0,255,80) if method=="blob" else (0,140,255)
    for pt in [left_pt, right_pt]:
        if pt:
            cv2.circle(vis, pt, 4, col, -1)
            cv2.circle(vis, pt, 7, col, 1)
    if left_pt and right_pt:
        cv2.line(vis, left_pt, right_pt, col, 1)
    status = f"{'BLOB' if method=='blob' else 'RATIO'}  f{idx:05d}"
    cv2.putText(vis, status, (5,18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
    return vis

# ── MAIN LOOP ──────────────────────────────────────────────────────────────────

def run(sample_every=1, preview_n=0):
    VIS_DIR.mkdir(exist_ok=True)
    cascade = make_cascade()

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        sys.exit(f"Cannot open {VIDEO_PATH}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if preview_n:
        total = min(total, preview_n)
    print(f"Video: {total} frames @ {fps:.1f} fps  |  sample_every={sample_every}")

    records   = []
    last_face : tuple | None = None
    face_history : list = []    # (frame_idx, face_box) for interpolation
    n_blob   = 0
    n_ratio  = 0
    n_noface = 0

    save_vis_every = max(1, total // (sample_every * 200))

    for fi in range(total):
        ret, frame = cap.read()
        if not ret:
            break

        if fi % sample_every != 0:
            continue

        timestamp = fi / fps

        # ── Face detection / tracking ──────────────────────────────────────────
        face_box = None
        if fi % HAAR_EVERY == 0:
            detected = detect_face(frame, cascade, last_face)
            if detected:
                last_face = detected
                face_history.append((fi, detected))
                if len(face_history) > 20:
                    face_history.pop(0)
                face_box = detected
            else:
                face_box = last_face
        else:
            face_box = last_face

        # ── Nostril finding ────────────────────────────────────────────────────
        left_pt = right_pt = None
        method  = "none"

        if face_box:
            pair = find_nostrils_blob(frame, face_box)
            if pair:
                left_pt, right_pt = pair
                method = "blob"
                n_blob += 1
            else:
                left_pt, right_pt = nostrils_from_ratios(face_box)
                method = "ratio"
                n_ratio += 1
        else:
            n_noface += 1

        # ── Thermal values ─────────────────────────────────────────────────────
        row = {
            "frame_idx":    fi,
            "timestamp_s":  round(timestamp, 3),
            "face_detected": face_box is not None,
            "method":        method,
        }
        if face_box:
            fx, fy, fw, fh = face_box
            row.update({"face_x":fx, "face_y":fy, "face_w":fw, "face_h":fh})

        for name, pt in [("left_nostril", left_pt), ("right_nostril", right_pt)]:
            if pt:
                row[f"{name}_px"]  = pt[0]
                row[f"{name}_py"]  = pt[1]
                row[f"{name}_TI"]  = round(patch_ti(frame, pt[0], pt[1]), 3)
                row[f"{name}_B"]   = round(patch_b(frame,  pt[0], pt[1]), 3)
            else:
                for sfx in ("px","py","TI","B"):
                    row[f"{name}_{sfx}"] = np.nan

        records.append(row)

        # ── Visualisation ──────────────────────────────────────────────────────
        seq = len(records)
        if seq % save_vis_every == 0:
            vis = draw_vis(frame, face_box, left_pt, right_pt, method, fi)
            cv2.imwrite(str(VIS_DIR / f"vis_{fi:06d}.jpg"), vis)

        if seq % 2000 == 0:
            total_so_far = n_blob + n_ratio + n_noface
            print(f"  {seq} frames  blob={n_blob} ratio={n_ratio} no-face={n_noface}")

    cap.release()

    df = pd.DataFrame(records)
    if not preview_n:
        df.to_csv(OUT_CSV, index=False)

    total_det = n_blob + n_ratio
    total_all = n_blob + n_ratio + n_noface
    print(f"\n{'-'*60}")
    print(f"Processed  : {len(df)} frames")
    print(f"Blob detect: {n_blob} ({100*n_blob/max(1,total_all):.1f}%)")
    print(f"Ratio fallb: {n_ratio} ({100*n_ratio/max(1,total_all):.1f}%)")
    print(f"No face    : {n_noface} ({100*n_noface/max(1,total_all):.1f}%)")

    det = df[df["face_detected"]]
    if len(det):
        print(f"\nLeft nostril  (orig coords):  "
              f"x={det.left_nostril_px.mean():.1f}±{det.left_nostril_px.std():.1f}  "
              f"y={det.left_nostril_py.mean():.1f}±{det.left_nostril_py.std():.1f}")
        print(f"Right nostril (orig coords):  "
              f"x={det.right_nostril_px.mean():.1f}±{det.right_nostril_px.std():.1f}  "
              f"y={det.right_nostril_py.mean():.1f}±{det.right_nostril_py.std():.1f}")
        print(f"Left  nostril B-channel:  mean={det.left_nostril_B.mean():.1f}  "
              f"std={det.left_nostril_B.std():.2f}")
        print(f"Right nostril B-channel:  mean={det.right_nostril_B.mean():.1f}  "
              f"std={det.right_nostril_B.std():.2f}")

    if not preview_n:
        print(f"\nCSV saved  : {OUT_CSV}")
    print(f"Vis frames : {VIS_DIR}/")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", type=int, default=0,
                    help="Process only first N frames (quick test)")
    ap.add_argument("--every",   type=int, default=1,
                    help="Sample every Nth frame (default 1 = all frames)")
    args = ap.parse_args()
    run(sample_every=args.every, preview_n=args.preview)
