import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np

VIDEO_PATH = r"NEEMA_Day1-001\NEEMA_Day1\backup_2025-12-26_12-53-29_NEEMA_DAY1\analysis_output_1080p.mp4"
MODEL_PATH = "face_landmarker.task"

NOSTRIL_LANDMARKS = {
    "nose_tip":            4,
    "left_nostril_outer":  129,
    "right_nostril_outer": 358,
    "left_nostril_inner":  218,
    "right_nostril_inner": 438,
    "left_ala_base":       166,
    "right_ala_base":      391,
}

# Read 10% frame — woman's face clearly visible here
cap = cv2.VideoCapture(VIDEO_PATH)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w_v = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h_v = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * 0.1))
ret, frame = cap.read()
cap.release()
h, w = frame.shape[:2]
print(f"Frame: {w}x{h}")

# Face region (right-half coords): y:320-530, x:450-750
# (located by visual inspection via grid/skin-detection)
x_off = w // 2
ry1, ry2, rx1, rx2 = 320, 530, 450, 750
face_crop_small = frame[ry1:ry2, x_off + rx1:x_off + rx2].copy()

# Upscale 3x for better landmark detection
scale = 3
face_crop = cv2.resize(face_crop_small, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_CUBIC)
ch, cw = face_crop.shape[:2]

# ── Run MediaPipe ─────────────────────────────────────────────────────────────
base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = mp_vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    min_face_detection_confidence=0.05,
    min_face_presence_confidence=0.05,
)
detector = mp_vision.FaceLandmarker.create_from_options(options)
mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                  data=cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
result = detector.detect(mp_img)

if not result.face_landmarks:
    raise RuntimeError("No landmarks detected. Check face_refined.png for content.")

face_lm = result.face_landmarks[0]

# ── Map landmarks back to original full-frame pixel coords ───────────────────
# Crop in full frame: x: (x_off+rx1) to (x_off+rx2), y: ry1 to ry2
# Upscaled by `scale`, so divide by scale to get crop-small coords, then add offsets
def to_full(lm):
    px = int((lm.x * cw) / scale) + x_off + rx1
    py = int((lm.y * ch) / scale) + ry1
    return px, py

print("\n=== Nostril / Nose Landmarks (pixel coords in original 1920x1080 frame) ===")
nostril_coords = {}
for name, idx in NOSTRIL_LANDMARKS.items():
    px, py = to_full(face_lm[idx])
    nostril_coords[name] = (px, py)
    print(f"  {name:25s}: ({px:4d}, {py:4d})")

# ── Annotate full frame ───────────────────────────────────────────────────────
annotated = frame.copy()

# Draw all 478 mesh points lightly
for lm in face_lm:
    px, py = to_full(lm)
    cv2.circle(annotated, (px, py), 1, (160, 160, 160), -1)

colors = {
    "nose_tip":            (0, 255, 0),
    "left_nostril_outer":  (0, 80, 255),
    "right_nostril_outer": (0, 80, 255),
    "left_nostril_inner":  (255, 80, 0),
    "right_nostril_inner": (255, 80, 0),
    "left_ala_base":       (0, 200, 200),
    "right_ala_base":      (0, 200, 200),
}
for name, (px, py) in nostril_coords.items():
    cv2.circle(annotated, (px, py), 7, colors[name], -1)
    cv2.putText(annotated, name, (px + 9, py - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, colors[name], 1)

# Face bounding box
cv2.rectangle(annotated, (x_off + rx1, ry1), (x_off + rx2, ry2), (0, 255, 255), 1)

# Nose zoom
pts = np.array(list(nostril_coords.values()))
nx1 = max(0, int(pts[:, 0].min()) - 20)
ny1 = max(0, int(pts[:, 1].min()) - 20)
nx2 = min(w - 1, int(pts[:, 0].max()) + 20)
ny2 = min(h - 1, int(pts[:, 1].max()) + 20)
cv2.rectangle(annotated, (nx1, ny1), (nx2, ny2), (255, 255, 0), 2)

nose_crop_raw = frame[ny1:ny2, nx1:nx2]
zoom = 5
nose_zoom = cv2.resize(nose_crop_raw, None, fx=zoom, fy=zoom,
                       interpolation=cv2.INTER_CUBIC)
for name, (px, py) in nostril_coords.items():
    zpx = (px - nx1) * zoom
    zpy = (py - ny1) * zoom
    cv2.circle(nose_zoom, (zpx, zpy), 7, colors[name], -1)
    cv2.putText(nose_zoom, name.replace("_", " "), (zpx + 8, zpy - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, colors[name], 1)

cv2.imwrite("woman_nostril_annotated.png", annotated)
cv2.imwrite("woman_face_crop.png", face_crop_small)
cv2.imwrite("woman_nose_zoom.png", nose_zoom)

print(f"\nSaved: woman_nostril_annotated.png  (full frame)")
print(f"Saved: woman_face_crop.png          (face region)")
print(f"Saved: woman_nose_zoom.png          (5x nose close-up)")
