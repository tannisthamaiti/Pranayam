"""
Manual nostril picker — interactive annotation tool for thermal video.

Left-click        : place the ACTIVE side (shown in HUD) — auto-toggles after each pick
Right-click       : undo / delete the active side's pick
Scroll wheel      : zoom in / zoom out (centred on cursor)
Middle-click drag : pan when zoomed in

Keyboard:
  l           : set active side to LEFT
  k           : set active side to RIGHT
  Space / n   : next frame
  p / b       : previous frame
  N (shift+n) : skip +10 frames
  P (shift+p) : skip -10 frames
  f           : go to specific frame number (type in terminal)
  c           : cycle contrast mode
  r           : reset zoom to full frame
  d           : delete ALL picks for current frame
  s           : save CSV now
  q / Esc     : save and quit

Outputs:
  manual_nostril_picks.csv  — frame_idx, timestamp_s, left/right px/py
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# ── CONFIG ─────────────────────────────────────────────────────────────────────
VIDEO_PATH = "NEEMA_Day1-001\\NEEMA_Day1\\backup_2025-12-26_12-53-29_NEEMA_DAY1\\check_thermal.mp4"
OUT_CSV    = "manual_nostril_picks.csv"

# Display scale applied to the raw frame size to get the window dimensions.
# The viewport zoom is on top of this — it is not affected by DISPLAY_SCALE.
DISPLAY_SCALE = 2
ZOOM_STEP     = 1.25   # factor per scroll notch

# ── CONTRAST MODES ─────────────────────────────────────────────────────────────

MODES = [
    "turbo",          # cool=blue/purple  warm=red/orange — best separation
    "coolness_blue",  # nostrils are BRIGHT CYAN on dark bg — easy to spot
    "magma",          # cool=dark-purple  warm=white/yellow — soft, no glare
    "viridis",        # cool=dark-blue    warm=yellow — perceptually uniform
    "inverted_gray",  # nostrils WHITE, skin dark — maximum contrast, no colour
    "original",
    "clahe_gray",
    "clahe_hot",
    "clahe_jet",
    "coolness_hot",
    "b_channel_clahe",
    "dog_bone",
]

def _clahe_gray(frame_bgr, clip=3.0, tile=8):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(gray)

def apply_contrast(frame_bgr, mode):
    b = frame_bgr[:, :, 0].astype(np.float32)
    g = frame_bgr[:, :, 1].astype(np.float32)
    r = frame_bgr[:, :, 2].astype(np.float32)

    if mode == "turbo":
        # Best for nostrils: cool spots appear distinct blue/purple vs warm red skin
        return cv2.applyColorMap(_clahe_gray(frame_bgr, clip=3.0), cv2.COLORMAP_TURBO)

    elif mode == "coolness_blue":
        # Invert TI so nostrils are the BRIGHTEST spots, then map to WINTER (cyan/blue)
        ti = r / 2 + g / 4 + b / 8
        coolness = np.clip(255.0 - ti, 0, 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(6, 6))
        return cv2.applyColorMap(clahe.apply(coolness), cv2.COLORMAP_WINTER)

    elif mode == "magma":
        # Soft gradient — cool=dark purple, warm=yellow/white, no harsh red
        return cv2.applyColorMap(_clahe_gray(frame_bgr, clip=3.0), cv2.COLORMAP_MAGMA)

    elif mode == "viridis":
        # Perceptually uniform — cool=dark blue, warm=yellow, clean boundaries
        return cv2.applyColorMap(_clahe_gray(frame_bgr, clip=3.0), cv2.COLORMAP_VIRIDIS)

    elif mode == "inverted_gray":
        # Nostrils appear WHITE (brightest), warm skin is dark — no colour confusion
        g8 = _clahe_gray(frame_bgr, clip=4.0)
        return cv2.cvtColor(cv2.bitwise_not(g8), cv2.COLOR_GRAY2BGR)

    elif mode == "original":
        return frame_bgr.copy()

    elif mode == "clahe_gray":
        return cv2.cvtColor(_clahe_gray(frame_bgr), cv2.COLOR_GRAY2BGR)

    elif mode == "clahe_hot":
        return cv2.applyColorMap(_clahe_gray(frame_bgr, clip=3.0), cv2.COLORMAP_HOT)

    elif mode == "clahe_jet":
        return cv2.applyColorMap(_clahe_gray(frame_bgr, clip=4.0), cv2.COLORMAP_JET)

    elif mode == "coolness_hot":
        ti = r / 2 + g / 4 + b / 8
        coolness = np.clip(255.0 - ti, 0, 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        return cv2.applyColorMap(clahe.apply(coolness), cv2.COLORMAP_HOT)

    elif mode == "b_channel_clahe":
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        b_enh = clahe.apply(frame_bgr[:, :, 0])
        out = frame_bgr.copy()
        out[:, :, 0] = b_enh
        return out

    elif mode == "dog_bone":
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        dog = cv2.GaussianBlur(gray, (21, 21), 0) - cv2.GaussianBlur(gray, (5, 5), 0)
        dog = np.clip(dog, 0, None)
        dog = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return cv2.applyColorMap(clahe.apply(dog), cv2.COLORMAP_BONE)

    return frame_bgr.copy()


# ── VIEWPORT (scroll-zoom + pan) ───────────────────────────────────────────────

class Viewport:
    """Tracks which rectangle of the original frame fills the display window."""

    def __init__(self, frame_w, frame_h):
        self.fw = frame_w
        self.fh = frame_h
        self.disp_w = frame_w  * DISPLAY_SCALE
        self.disp_h = frame_h * DISPLAY_SCALE
        self.reset()

    def reset(self):
        self.vx = 0.0
        self.vy = 0.0
        self.vw = float(self.fw)
        self.vh = float(self.fh)

    def scroll_zoom(self, factor, mx, my):
        """Zoom in (factor<1) or out (factor>1) keeping cursor position fixed."""
        new_w = max(10.0, min(self.fw, self.vw * factor))
        new_h = max(10.0, min(self.fh, self.vh * factor))
        fx = self.vx + mx * self.vw / self.disp_w
        fy = self.vy + my * self.vh / self.disp_h
        self.vx = fx - mx * new_w / self.disp_w
        self.vy = fy - my * new_h / self.disp_h
        self.vw = new_w
        self.vh = new_h
        self._clamp()

    def pan(self, ddx, ddy):
        """Pan by ddx/ddy display pixels."""
        self.vx -= ddx * self.vw / self.disp_w
        self.vy -= ddy * self.vh / self.disp_h
        self._clamp()

    def _clamp(self):
        self.vx = max(0.0, min(self.fw - self.vw, self.vx))
        self.vy = max(0.0, min(self.fh - self.vh, self.vy))

    def to_frame(self, dx, dy):
        """Display pixel → original frame pixel."""
        return (
            int(self.vx + dx * self.vw / self.disp_w),
            int(self.vy + dy * self.vh / self.disp_h),
        )

    def to_display(self, ox, oy):
        """Original frame pixel → display pixel."""
        return (
            int((ox - self.vx) * self.disp_w / self.vw),
            int((oy - self.vy) * self.disp_h / self.vh),
        )

    def crop_and_resize(self, img):
        x1 = int(np.clip(self.vx, 0, self.fw))
        y1 = int(np.clip(self.vy, 0, self.fh))
        x2 = int(np.clip(self.vx + self.vw, 0, self.fw))
        y2 = int(np.clip(self.vy + self.vh, 0, self.fh))
        crop = img[y1:y2, x1:x2]
        interp = cv2.INTER_LINEAR if self.vw < self.fw * 0.9 else cv2.INTER_NEAREST
        return cv2.resize(crop, (self.disp_w, self.disp_h), interpolation=interp)

    @property
    def zoom_level(self):
        return self.fw / self.vw


# ── DISPLAY ────────────────────────────────────────────────────────────────────

SIDE_COL = {"left": (0, 255, 80), "right": (60, 60, 255)}

def make_display(frame_bgr, frame_idx, fps, total, picks, mode, vp, next_side):
    """Build the BGR image shown in the window using the given Viewport."""
    vis = apply_contrast(frame_bgr, mode)
    vis = vp.crop_and_resize(vis)
    dh, dw = vis.shape[:2]

    # Draw nostril picks
    pick = picks.get(frame_idx, {})
    for side, col in SIDE_COL.items():
        pt = pick.get(side)
        if pt:
            dx, dy = vp.to_display(*pt)
            cv2.circle(vis, (dx, dy), 7, col, -1)
            cv2.circle(vis, (dx, dy), 11, col, 1)
            cv2.putText(vis, side[0].upper(), (dx + 13, dy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)

    if pick.get("left") and pick.get("right"):
        ld = vp.to_display(*pick["left"])
        rd = vp.to_display(*pick["right"])
        cv2.line(vis, ld, rd, (255, 255, 255), 1)

    # ── Active-side badge (top-right, hard to miss) ───────────────────────────
    act_col  = SIDE_COL[next_side]
    act_done = next_side in pick
    badge    = f"{'[DONE] ' if act_done else ''}CLICK = {next_side.upper()}"
    (tw, th), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    bx = dw - tw - 10
    by = 28
    cv2.rectangle(vis, (bx - 6, by - th - 4), (bx + tw + 4, by + 4),
                  (0, 0, 0), -1)
    cv2.putText(vis, badge, (bx, by), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, act_col, 2, cv2.LINE_AA)

    # ── Bottom HUD ────────────────────────────────────────────────────────────
    ts         = frame_idx / fps
    annotated  = sum(1 for p in picks.values() if p.get("left") and p.get("right"))
    both       = bool(pick.get("left") and pick.get("right"))
    zoom_str   = f"{vp.zoom_level:.1f}x" if vp.zoom_level > 1.05 else "1x"
    status_col = (100, 255, 100) if both else (80, 180, 255)

    lines = [
        (f"Frame {frame_idx}/{total-1}  {ts:.2f}s  [{mode}]  zoom:{zoom_str}", (180, 255, 180)),
        (f"Annotated: {annotated}  |  This frame: {'DONE' if both else 'INCOMPLETE'}", status_col),
        ("l=LEFT  k=RIGHT  R-click=undo  scroll=zoom  mid-drag=pan  r=reset-zoom", (200, 200, 200)),
        ("Space/n=next  p=prev  N/P=±10  c=mode  d=del-all  s=save  q=quit", (160, 160, 160)),
    ]
    for i, (txt, col) in enumerate(lines):
        y = 16 + i * 18
        cv2.putText(vis, txt, (6, y + 1), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(vis, txt, (6, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, col, 1, cv2.LINE_AA)

    return vis


# ── FRAME CACHE ────────────────────────────────────────────────────────────────

class FrameCache:
    def __init__(self, cap, max_size=60):
        self.cap      = cap
        self.cache    = {}
        self.max_size = max_size

    def get(self, idx):
        if idx in self.cache:
            return self.cache[idx]
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if not ret:
            return None
        self.cache[idx] = frame
        if len(self.cache) > self.max_size:
            oldest = min(self.cache)
            del self.cache[oldest]
        return frame


# ── MAIN ───────────────────────────────────────────────────────────────────────

def save_csv(picks, fps):
    rows = []
    for fi, p in sorted(picks.items()):
        lp = p.get("left")
        rp = p.get("right")
        rows.append({
            "frame_idx":        fi,
            "timestamp_s":      round(fi / fps, 3),
            "left_nostril_px":  lp[0] if lp else None,
            "left_nostril_py":  lp[1] if lp else None,
            "right_nostril_px": rp[0] if rp else None,
            "right_nostril_py": rp[1] if rp else None,
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"  Saved {len(df)} rows → {OUT_CSV}")
    return df


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        sys.exit(f"Cannot open {VIDEO_PATH}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    print(f"Loaded: {total} frames @ {fps:.1f} fps")
    print(f"Output: {OUT_CSV}")

    # Resume from existing CSV
    picks = {}
    if Path(OUT_CSV).exists():
        try:
            df_ex = pd.read_csv(OUT_CSV)
            for _, row in df_ex.iterrows():
                fi = int(row["frame_idx"])
                picks[fi] = {}
                if pd.notna(row.get("left_nostril_px")):
                    picks[fi]["left"]  = (int(row["left_nostril_px"]),
                                          int(row["left_nostril_py"]))
                if pd.notna(row.get("right_nostril_px")):
                    picks[fi]["right"] = (int(row["right_nostril_px"]),
                                          int(row["right_nostril_py"]))
            print(f"Resumed: {len(picks)} existing picks loaded.")
        except Exception as e:
            print(f"Warning: could not load existing CSV ({e}), starting fresh.")

    cache      = FrameCache(cap)
    frame_idx  = 0
    mode_idx   = 0
    next_side  = "left"   # which nostril the next click will place
    dirty      = True
    vp         = None     # Viewport — created once first frame dimensions are known

    click = {"x": -1, "y": -1, "btn": -1, "new": False}
    pan   = {"active": False, "last_x": 0, "last_y": 0}

    WIN = "Nostril Picker  —  click=place  l/k=switch side  scroll=zoom  q=quit"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    def on_mouse(event, x, y, flags, param):
        nonlocal dirty
        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            click.update(x=x, y=y, btn=0 if event == cv2.EVENT_LBUTTONDOWN else 1, new=True)
        elif event == cv2.EVENT_MOUSEWHEEL and vp is not None:
            vp.scroll_zoom((1.0 / ZOOM_STEP) if flags > 0 else ZOOM_STEP, x, y)
            dirty = True
        elif event == cv2.EVENT_MBUTTONDOWN:
            pan.update(active=True, last_x=x, last_y=y)
        elif event == cv2.EVENT_MOUSEMOVE and pan["active"] and vp is not None:
            vp.pan(x - pan["last_x"], y - pan["last_y"])
            pan["last_x"] = x; pan["last_y"] = y
            dirty = True
        elif event == cv2.EVENT_MBUTTONUP:
            pan["active"] = False

    cv2.setMouseCallback(WIN, on_mouse)

    while True:
        # ── Render ────────────────────────────────────────────────────────────
        if dirty:
            frame = cache.get(frame_idx)
            if frame is None:
                frame_idx = max(0, frame_idx - 1)
                continue
            if vp is None:
                fh, fw = frame.shape[:2]
                vp = Viewport(fw, fh)
                cv2.resizeWindow(WIN, vp.disp_w, vp.disp_h)
            mode_name = MODES[mode_idx % len(MODES)]
            vis = make_display(frame, frame_idx, fps, total, picks,
                               mode_name, vp, next_side)
            cv2.imshow(WIN, vis)
            dirty = False

        # ── Nostril click ─────────────────────────────────────────────────────
        if click["new"]:
            click["new"] = False
            ox, oy = vp.to_frame(click["x"], click["y"])

            if click["btn"] == 0:                    # left-click → place active side
                picks.setdefault(frame_idx, {})[next_side] = (ox, oy)
                print(f"  f{frame_idx:05d}  {next_side:5s} → ({ox}, {oy})")
                next_side = "right" if next_side == "left" else "left"
            else:                                    # right-click → undo active side
                picks.get(frame_idx, {}).pop(next_side, None)
                print(f"  f{frame_idx:05d}  {next_side} removed")
            dirty = True

        # ── Keyboard ──────────────────────────────────────────────────────────
        raw = cv2.waitKey(20)
        if raw == -1:
            continue
        key = raw & 0xFF

        if key in (ord('q'), 27):           # q / Esc — save and quit
            save_csv(picks, fps)
            break

        elif key == ord('s'):               # save
            save_csv(picks, fps)

        elif key == ord('l'):               # force LEFT side active
            next_side = "left"
            dirty = True

        elif key == ord('k'):               # force RIGHT side active
            next_side = "right"
            dirty = True

        elif key == ord('c'):               # cycle contrast
            mode_idx += 1
            print(f"  Mode: {MODES[mode_idx % len(MODES)]}")
            dirty = True

        elif key == ord('r') and vp is not None:   # reset zoom
            vp.reset()
            dirty = True

        elif key == ord('d'):               # delete picks
            picks.pop(frame_idx, None)
            print(f"  f{frame_idx:05d}  picks deleted")
            dirty = True

        elif key == ord('f'):               # go to frame
            try:
                target = int(input(f"  Go to frame (0–{total-1}): ").strip())
                frame_idx = max(0, min(total - 1, target))
                dirty = True
            except (ValueError, EOFError):
                pass

        elif key in (ord(' '), ord('n')):   # next frame
            frame_idx = min(total - 1, frame_idx + 1)
            dirty = True

        elif key in (ord('p'), ord('b')):   # previous frame
            frame_idx = max(0, frame_idx - 1)
            dirty = True

        elif key == ord('N'):               # skip +10
            frame_idx = min(total - 1, frame_idx + 10)
            dirty = True

        elif key == ord('P'):               # skip -10
            frame_idx = max(0, frame_idx - 10)
            dirty = True

        # Windows arrow keys (raw values, no 0xFF mask)
        elif raw == 2555904:                # right arrow
            frame_idx = min(total - 1, frame_idx + 1)
            dirty = True
        elif raw == 2424832:                # left arrow
            frame_idx = max(0, frame_idx - 1)
            dirty = True
        elif raw == 2490368:                # up arrow — skip +10
            frame_idx = min(total - 1, frame_idx + 10)
            dirty = True
        elif raw == 2621440:                # down arrow — skip -10
            frame_idx = max(0, frame_idx - 10)
            dirty = True

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
