import cv2
import os
import numpy as np

# ── CONFIG ────────────────────────────────────────────────
IMAGE_PATH = "tracking_frames/frame_000000.jpg"   # change if needed
OUTPUT_IMG = "debug_landmarks.jpg"

# ── STATE ─────────────────────────────────────────────────
clicked_points = []
face_box       = None
base_frame     = None

# Zoom state
zoom_level  = 1.0
zoom_center = None   # (cx, cy) in original image coords
ZOOM_STEP   = 0.25
ZOOM_MIN    = 1.0
ZOOM_MAX    = 8.0

POINT_NAMES = [
    "nose_bridge",
    "nose_tip",
    "left_nostril",
    "right_nostril",
    "nose_left_ala",
    "nose_right_ala",
    "philtrum",
    "nose_base_left",
    "nose_base_right",
]

COLORS = [
    (0,255,0),(0,200,255),(255,100,0),(0,100,255),
    (200,0,200),(0,255,200),(255,255,0),(100,255,100),(255,100,100),
]

WIN = "Landmark Picker  |  Scroll=zoom  Click=place  U=undo  R=reset  S=save  Q=quit"


def view_to_img(vx, vy, frame_shape):
    """Convert zoomed view coords to original image coords."""
    ih, iw = frame_shape[:2]
    cx, cy = zoom_center if zoom_center else (iw/2, ih/2)

    half_w = iw / (2 * zoom_level)
    half_h = ih / (2 * zoom_level)

    x0 = cx - half_w
    y0 = cy - half_h

    ix = x0 + vx / zoom_level
    iy = y0 + vy / zoom_level
    return int(np.clip(ix, 0, iw-1)), int(np.clip(iy, 0, ih-1))


def get_zoomed_view(frame):
    """Crop and resize frame based on zoom_level and zoom_center."""
    ih, iw = frame.shape[:2]
    cx, cy = zoom_center if zoom_center else (iw/2, ih/2)

    half_w = iw / (2 * zoom_level)
    half_h = ih / (2 * zoom_level)

    x0 = int(np.clip(cx - half_w, 0, iw))
    y0 = int(np.clip(cy - half_h, 0, ih))
    x1 = int(np.clip(cx + half_w, 0, iw))
    y1 = int(np.clip(cy + half_h, 0, ih))

    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return frame.copy()
    return cv2.resize(crop, (iw, ih), interpolation=cv2.INTER_LINEAR)


def redraw():
    vis = base_frame.copy()

    # Draw face bounding box
    if face_box:
        x, y, w, h = face_box
        cv2.rectangle(vis, (x, y), (x+w, y+h), (255, 255, 0), 2)

    # Draw placed points on original-coord image
    for i, (name, px, py, rx, ry) in enumerate(clicked_points):
        color = COLORS[i % len(COLORS)]
        cv2.circle(vis, (px, py), 5, color, -1)
        cv2.putText(vis, f"{i+1}:{name}", (px+6, py),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Apply zoom
    view = get_zoomed_view(vis)

    # HUD on view layer
    idx = len(clicked_points)
    if idx < len(POINT_NAMES):
        msg = f"Click: {POINT_NAMES[idx]}  ({idx+1}/{len(POINT_NAMES)})"
        cv2.putText(view, msg, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    else:
        cv2.putText(view, "All points placed!  Press S to save",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 100), 2)

    cv2.putText(view, f"Zoom: {zoom_level:.1f}x  (scroll to zoom)",
                (10, view.shape[0]-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow(WIN, view)


def mouse_cb(event, vx, vy, flags, param):
    global zoom_level, zoom_center

    # ── Scroll wheel → zoom ───────────────────────────────
    if event == cv2.EVENT_MOUSEWHEEL:
        ix, iy = view_to_img(vx, vy, base_frame.shape)

        if flags > 0:
            zoom_level = min(zoom_level + ZOOM_STEP, ZOOM_MAX)
        else:
            zoom_level = max(zoom_level - ZOOM_STEP, ZOOM_MIN)

        if zoom_level == ZOOM_MIN:
            ih, iw = base_frame.shape[:2]
            zoom_center = (iw / 2, ih / 2)
        else:
            zoom_center = (float(ix), float(iy))

        redraw()
        return

    # ── Left click → place landmark ───────────────────────
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_points) >= len(POINT_NAMES):
            print("All points placed. Press U to undo or R to reset.")
            return

        ix, iy = view_to_img(vx, vy, base_frame.shape)

        rx, ry = None, None
        if face_box:
            x, y, w, h = face_box
            rx = round((ix - x) / w, 4)
            ry = round((iy - y) / h, 4)

        name = POINT_NAMES[len(clicked_points)]
        clicked_points.append((name, ix, iy, rx, ry))
        print(f"  [{len(clicked_points)}] {name:20s}  img=({ix},{iy})  ratio=({rx}, {ry})")
        redraw()


def print_summary():
    print("\n── NOSE_REGIONS dict (paste into tracker) ──────────────")
    print("NOSE_REGIONS = {")
    for name, px, py, rx, ry in clicked_points:
        print(f'    "{name}": ({rx}, {ry}),')
    print("}")


def main():
    global base_frame, face_box, zoom_center

    if not os.path.exists(IMAGE_PATH):
        print(f"Image not found: {IMAGE_PATH}")
        return

    frame = cv2.imread(IMAGE_PATH)
    if frame is None:
        print("Could not read image.")
        return

    ih, iw = frame.shape[:2]
    zoom_center = (iw / 2, ih / 2)

    # Detect face
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    r_chan = frame[:, :, 2]
    faces  = face_cascade.detectMultiScale(r_chan, 1.1, 4, minSize=(60, 60))

    if len(faces) == 0:
        print("No face detected — ratios won't be calculated.")
    else:
        face_box = tuple(sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0])
        x, y, w, h = face_box
        print(f"Face detected at x={x} y={y} w={w} h={h}")
        zoom_center = (x + w / 2.0, y + h / 2.0)

    base_frame = frame.copy()

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 900, 700)
    cv2.setMouseCallback(WIN, mouse_cb)

    print("\n── Controls ─────────────────────────────────────────────")
    print("  Scroll wheel : zoom in / out (centered on cursor)")
    print("  Left click   : place next landmark")
    print("  U            : undo last point")
    print("  R            : reset all points")
    print("  S            : save image + print NOSE_REGIONS dict")
    print("  Q            : quit")
    print("\n── Point order ──────────────────────────────────────────")
    for i, n in enumerate(POINT_NAMES):
        print(f"  {i+1:2d}. {n}")
    print()

    redraw()

    while True:
        key = cv2.waitKey(20) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('u'):
            if clicked_points:
                removed = clicked_points.pop()
                print(f"  Undone: {removed[0]}")
                redraw()

        elif key == ord('r'):
            clicked_points.clear()
            print("  Reset all points.")
            redraw()

        elif key == ord('s'):
            # Save full-res image (no zoom)
            out = base_frame.copy()
            if face_box:
                x, y, w, h = face_box
                cv2.rectangle(out, (x, y), (x+w, y+h), (255, 255, 0), 2)
            for i, (name, px, py, rx, ry) in enumerate(clicked_points):
                color = COLORS[i % len(COLORS)]
                cv2.circle(out, (px, py), 5, color, -1)
                cv2.putText(out, name, (px+6, py),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            cv2.imwrite(OUTPUT_IMG, out)
            print(f"\n  Saved: {OUTPUT_IMG}")
            print_summary()

    cv2.destroyAllWindows()
    if clicked_points:
        print_summary()


if __name__ == "__main__":
    main()