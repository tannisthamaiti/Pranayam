import cv2
import numpy as np
import os

VIDEO_PATH = r"NEEMA_Day1-001\NEEMA_Day1\backup_2025-12-26_12-53-29_NEEMA_DAY1\analysis_output_1080p.mp4"
OUT_DIR = "woman_face_frames"
os.makedirs(OUT_DIR, exist_ok=True)

# Known face region in the full 1920x1080 frame (right half, located by visual inspection)
# Full frame coords: x: 1410-1710, y: 320-530
FACE_X1, FACE_X2 = 1410, 1710
FACE_Y1, FACE_Y2 = 320, 530

# Nose zoom region within the face crop (approximate centre of face crop)
# Face crop size: 300x210 — nose is roughly in the middle vertically, centre horizontally
# Adjust these offsets relative to face crop top-left to zoom onto the nose/nostril area
NOSE_X1, NOSE_X2 = 70, 230   # within face crop (300px wide)
NOSE_Y1, NOSE_Y2 = 60, 160   # within face crop (210px high)

ZOOM = 4   # magnification for nose zoom panel

cap = cv2.VideoCapture(VIDEO_PATH)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps   = cap.get(cv2.CAP_PROP_FPS)
print(f"Video: {total} frames @ {fps:.1f} fps  (~{total/fps:.0f}s)")
print(f"Saving to: {OUT_DIR}/")

saved = 0
fi = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Face crop
    face = frame[FACE_Y1:FACE_Y2, FACE_X1:FACE_X2]

    # Nose zoom — crop from face, upscale
    nose_small = face[NOSE_Y1:NOSE_Y2, NOSE_X1:NOSE_X2]
    nose_zoom  = cv2.resize(nose_small, None, fx=ZOOM, fy=ZOOM,
                            interpolation=cv2.INTER_CUBIC)

    # Stack face crop (resized to same height as nose zoom) and nose zoom side by side
    nz_h, nz_w = nose_zoom.shape[:2]
    face_resized = cv2.resize(face, (int(face.shape[1] * nz_h / face.shape[0]), nz_h))

    # Add labels
    face_labeled = face_resized.copy()
    cv2.putText(face_labeled, f"frame {fi:05d}", (5, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    # Draw nose zoom region on face panel
    sx = int(NOSE_X1 * face_resized.shape[1] / face.shape[1])
    ex = int(NOSE_X2 * face_resized.shape[1] / face.shape[1])
    sy = int(NOSE_Y1 * nz_h / face.shape[0])
    ey = int(NOSE_Y2 * nz_h / face.shape[0])
    cv2.rectangle(face_labeled, (sx, sy), (ex, ey), (0, 255, 255), 1)

    nose_labeled = nose_zoom.copy()
    cv2.putText(nose_labeled, "nose zoom", (4, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    combined = np.hstack([face_labeled, nose_labeled])
    out_path = os.path.join(OUT_DIR, f"frame_{fi:05d}.jpg")
    cv2.imwrite(out_path, combined, [cv2.IMWRITE_JPEG_QUALITY, 90])

    saved += 1
    if saved % 500 == 0:
        print(f"  {saved}/{total} frames saved...")

    fi += 1

cap.release()
print(f"\nDone. {saved} frames saved to '{OUT_DIR}/'")
