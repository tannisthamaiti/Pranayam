"""
Manual nostril labeler for thermal face frames.

Shows frames from thermal_face_frames/ one at a time.
Left panel = face overview, right panel = nose zoom (6x).
Click directly on the nose zoom panel for best accuracy.

Controls:
  Left click (1st) : place LEFT nostril
  Left click (2nd) : place RIGHT nostril
  U                : undo last point
  R                : redo / clear both points this frame
  S or N           : skip (nostrils not visible / occluded)
  Space / Enter    : accept & next frame
  B                : go back one frame
  Q                : quit & save

Outputs:
  nostril_labels.csv   — saved after every accepted frame

Coordinate columns:
  panel_x / panel_y    : pixels in the nose zoom panel  (0-780, 0-540)
  orig_x  / orig_y     : pixels in original 640x480 thermal frame
                         orig_x = panel_x / 6 + 260
                         orig_y = panel_y / 6 + 145
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# ── CONFIG ─────────────────────────────────────────────────────────────────────
FRAMES_DIR  = Path("thermal_face_frames")
OUT_CSV     = "nostril_labels.csv"

# Composite frame layout
FACE_PANEL_W = 675          # left panel width
NOSE_PANEL_W = 780          # right panel width (6x zoom)
FRAME_H      = 540

# Original thermal frame nose region origin
NOSE_ORIG_X  = 260
NOSE_ORIG_Y  = 145
ZOOM_FACTOR  = 6

# Step between frames shown (100 = ~354 frames from 35k total)
FRAME_STEP   = 100

# Display: downscale composite to fit screen (1455 wide is large)
DISP_SCALE   = 0.85         # set lower if window is too wide

# ── COORDINATE HELPERS ─────────────────────────────────────────────────────────

def panel_to_orig(px, py):
    """Nose panel pixel → original 640x480 thermal frame coords."""
    return round(px / ZOOM_FACTOR + NOSE_ORIG_X, 2), \
           round(py / ZOOM_FACTOR + NOSE_ORIG_Y, 2)

def click_to_panel(click_x, click_y, scale):
    """Convert display click → nose panel coords (subtract face panel offset)."""
    img_x = click_x / scale
    img_y = click_y / scale
    panel_x = img_x - FACE_PANEL_W   # subtract left panel width
    panel_y = img_y
    return panel_x, panel_y

def in_nose_panel(click_x, click_y, scale):
    """True if click falls within the nose zoom panel."""
    img_x = click_x / scale
    return img_x >= FACE_PANEL_W

# ── STATE ──────────────────────────────────────────────────────────────────────

clicks   = []    # list of (panel_x, panel_y) — max 2
WIN      = "Nostril Labeler  |  Click L then R in NOSE PANEL  |  U=undo R=redo S=skip Space=next B=back Q=quit"

# ── DRAW ───────────────────────────────────────────────────────────────────────

COLORS = [(255, 80, 0), (0, 80, 255)]   # blue=left, red=right
LABELS = ["LEFT", "RIGHT"]


def draw(base_disp, frame_num, total, frame_idx, scale):
    vis  = base_disp.copy()
    dh, dw = vis.shape[:2]

    # Separator line between face and nose panels
    sep = int(FACE_PANEL_W * scale)
    cv2.line(vis, (sep, 0), (sep, dh), (80, 80, 80), 1)

    # Label panels
    cv2.putText(vis, "FACE OVERVIEW", (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120,120,120), 1)
    cv2.putText(vis, "NOSE ZOOM (6x)  <- click here", (sep+8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120,255,120), 1)

    # Placed clicks
    for i, (px, py) in enumerate(clicks):
        # Map panel coords → display coords
        dx = int((px + FACE_PANEL_W) * scale)
        dy = int(py * scale)
        col = COLORS[i]
        cv2.circle(vis, (dx, dy), 8, col, -1)
        cv2.circle(vis, (dx, dy), 10, (255,255,255), 1)
        cv2.putText(vis, LABELS[i], (dx+12, dy+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)

    # Status / instruction
    if len(clicks) == 0:
        msg = "Click LEFT nostril in nose panel"
    elif len(clicks) == 1:
        msg = "Now click RIGHT nostril"
    else:
        msg = "Both placed — Space/Enter to accept"

    cv2.putText(vis, msg, (sep+8, dh-30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
    cv2.putText(vis, f"Frame {frame_num}/{total}  (idx {frame_idx})",
                (8, dh-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
    cv2.putText(vis, "S=skip  U=undo  R=redo  B=back  Q=quit",
                (8, dh-10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150,150,150), 1)
    return vis


def mouse_cb(event, dx, dy, flags, param):
    scale = param["scale"]
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicks) >= 2:
            return
        if not in_nose_panel(dx, dy, scale):
            print("  Click inside the NOSE ZOOM panel (right half).")
            return
        px, py = click_to_panel(dx, dy, scale)
        # Clamp to panel bounds
        px = max(0, min(NOSE_PANEL_W - 1, px))
        py = max(0, min(FRAME_H - 1, py))
        clicks.append((px, py))
        ox, oy = panel_to_orig(px, py)
        name = LABELS[len(clicks)-1]
        print(f"  {name}: panel=({px:.0f},{py:.0f})  orig=({ox:.1f},{oy:.1f})")

# ── LOAD & SAVE ────────────────────────────────────────────────────────────────

def load_existing():
    p = Path(OUT_CSV)
    if p.exists():
        df = pd.read_csv(p)
        print(f"Resuming — {len(df)} frames already labelled in {OUT_CSV}")
        return df
    return pd.DataFrame()

def append_row(df, row):
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(OUT_CSV, index=False)
    return df

# ── MAIN ───────────────────────────────────────────────────────────────────────

def main():
    global clicks

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=FRAME_STEP,
                    help=f"Show every Nth frame (default {FRAME_STEP}). "
                         "Lower = more labels, higher = faster session.")
    args = ap.parse_args()

    frame_files = sorted(FRAMES_DIR.glob("frame_*.jpg"))
    if not frame_files:
        sys.exit(f"No frames found in {FRAMES_DIR}/")

    # Select frames to label
    selected = frame_files[::args.step]
    total    = len(selected)
    print(f"Found {len(frame_files)} frames, showing every {FRAME_STEP}th = {total} to label")

    df = load_existing()
    done_idxs = set(df["frame_idx"].tolist()) if len(df) else set()

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    i = 0
    while i < total:
        fp = selected[i]
        frame_idx = int(fp.stem.split("_")[1])

        # Skip already-labelled frames
        if frame_idx in done_idxs:
            i += 1
            continue

        img = cv2.imread(str(fp))
        if img is None:
            i += 1
            continue

        scale = DISP_SCALE
        dw    = int(img.shape[1] * scale)
        dh    = int(img.shape[0] * scale)
        cv2.resizeWindow(WIN, dw, dh)
        base_disp = cv2.resize(img, (dw, dh), interpolation=cv2.INTER_LINEAR)

        clicks = []
        cv2.setMouseCallback(WIN, mouse_cb, {"scale": scale})

        print(f"\n[{i+1}/{total}] frame_{frame_idx:05d}.jpg")

        action = None
        while True:
            vis = draw(base_disp, i+1, total, frame_idx, scale)
            cv2.imshow(WIN, vis)
            key = cv2.waitKey(20) & 0xFF

            if key == ord('u') or key == ord('U'):
                if clicks:
                    removed = clicks.pop()
                    print(f"  Undo: removed {LABELS[len(clicks)]}")

            elif key == ord('r') or key == ord('R'):
                clicks = []
                print("  Reset.")

            elif key == ord('s') or key == ord('S') or key == ord('n') or key == ord('N'):
                action = "skip"
                break

            elif key == ord('b') or key == ord('B'):
                action = "back"
                break

            elif key == ord('q') or key == ord('Q'):
                action = "quit"
                break

            elif key in (ord(' '), 13):   # space or enter
                if len(clicks) == 2:
                    action = "accept"
                    break
                elif len(clicks) == 0:
                    action = "skip"
                    break
                else:
                    print("  Place both nostrils first (or S to skip).")

        if action == "quit":
            print("Quitting — labels saved.")
            break

        if action == "back":
            i = max(0, i - 1)
            # Remove previously saved label for that frame if any
            prev_idx = int(selected[i].stem.split("_")[1])
            if len(df) and prev_idx in df["frame_idx"].values:
                df = df[df["frame_idx"] != prev_idx]
                df.to_csv(OUT_CSV, index=False)
                done_idxs.discard(prev_idx)
            continue

        if action == "skip":
            row = {"frame_idx": frame_idx, "frame_file": fp.name,
                   "skipped": True,
                   "left_panel_x": None, "left_panel_y": None,
                   "right_panel_x": None, "right_panel_y": None,
                   "left_orig_x": None, "left_orig_y": None,
                   "right_orig_x": None, "right_orig_y": None}
            df = append_row(df, row)
            done_idxs.add(frame_idx)
            print(f"  Skipped.")
            i += 1
            continue

        if action == "accept":
            lx, ly = clicks[0]
            rx, ry = clicks[1]
            lox, loy = panel_to_orig(lx, ly)
            rox, roy = panel_to_orig(rx, ry)
            row = {"frame_idx": frame_idx, "frame_file": fp.name,
                   "skipped": False,
                   "left_panel_x":  round(lx, 1),  "left_panel_y":  round(ly, 1),
                   "right_panel_x": round(rx, 1),  "right_panel_y": round(ry, 1),
                   "left_orig_x":   lox,            "left_orig_y":   loy,
                   "right_orig_x":  rox,            "right_orig_y":  roy}
            df = append_row(df, row)
            done_idxs.add(frame_idx)
            lp = clicks[0]; rp = clicks[1]
            sep = abs(rp[0]-lp[0]) / ZOOM_FACTOR
            print(f"  Saved — sep={sep:.1f}px in orig frame")
            i += 1

    cv2.destroyAllWindows()

    labelled = df[df["skipped"]==False] if len(df) else pd.DataFrame()
    skipped  = df[df["skipped"]==True]  if len(df) else pd.DataFrame()
    print(f"\nDone. CSV: {OUT_CSV}")
    print(f"  Labelled : {len(labelled)}")
    print(f"  Skipped  : {len(skipped)}")
    if len(labelled):
        print(f"  Left  nostril orig: x={labelled.left_orig_x.mean():.1f}  y={labelled.left_orig_y.mean():.1f}")
        print(f"  Right nostril orig: x={labelled.right_orig_x.mean():.1f}  y={labelled.right_orig_y.mean():.1f}")


if __name__ == "__main__":
    main()
