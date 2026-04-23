import cv2
import numpy as np
import os

VIDEO_PATH = r"check_thermal.mp4"
OUT_DIR    = "thermal_face_frames"
os.makedirs(OUT_DIR, exist_ok=True)

# Crop regions in the 640x480 thermal frame
FACE_X1, FACE_X2, FACE_Y1, FACE_Y2 = 200, 450, 50, 250
NOSE_X1, NOSE_X2, NOSE_Y1, NOSE_Y2 = 260, 390, 145, 235
ZOOM = 6

cap   = cv2.VideoCapture(VIDEO_PATH)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps   = cap.get(cv2.CAP_PROP_FPS)
print(f"Video: {total} frames @ {fps:.1f} fps  (~{total/fps:.0f}s)")
print(f"Saving to: {OUT_DIR}/")

# Precompute nose box offset within face crop
nfx1 = NOSE_X1 - FACE_X1;  nfx2 = NOSE_X2 - FACE_X1
nfy1 = NOSE_Y1 - FACE_Y1;  nfy2 = NOSE_Y2 - FACE_Y1

saved = 0
fi    = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    face      = frame[FACE_Y1:FACE_Y2, FACE_X1:FACE_X2].copy()
    nose_zoom = cv2.resize(frame[NOSE_Y1:NOSE_Y2, NOSE_X1:NOSE_X2],
                           None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_CUBIC)

    # Draw nose box on face panel
    cv2.rectangle(face, (nfx1, nfy1), (nfx2, nfy2), (255, 255, 255), 1)

    # Frame label
    cv2.putText(face, f"frame {fi:05d}", (3, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # Resize face panel to match nose zoom height
    nz_h = nose_zoom.shape[0]
    face_r = cv2.resize(face, (int(face.shape[1] * nz_h / face.shape[0]), nz_h))

    combined = np.hstack([face_r, nose_zoom])
    cv2.imwrite(os.path.join(OUT_DIR, f"frame_{fi:05d}.jpg"),
                combined, [cv2.IMWRITE_JPEG_QUALITY, 92])

    saved += 1
    if saved % 1000 == 0:
        print(f"  {saved}/{total} frames saved...")

    fi += 1

cap.release()
print(f"\nDone. {saved} frames saved to '{OUT_DIR}/'")
