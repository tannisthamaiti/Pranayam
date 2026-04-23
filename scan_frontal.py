"""
Scan for woman's face using OpenCV Haarcascades (frontal + profile).
Also checks whether the existing skeleton dots on her face can be used.
"""
import cv2
import numpy as np

VIDEO_PATH = r"NEEMA_Day1-002\NEEMA_Day1\backup_2025-12-26_12-53-29_NEEMA_DAY1\3D_analysis_output_1080p.mp4"

# Load cascades
cascade_frontal = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cascade_profile = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")

cap = cv2.VideoCapture(VIDEO_PATH)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w_v = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h_v = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video: {w_v}x{h_v}, {total} frames")

best_hits = []  # (frame_idx, frame, faces, mode)

step = 15  # check every 15 frames
for fi in range(0, total, step):
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ret, frame = cap.read()
    if not ret:
        continue

    # Focus on right half (woman)
    right = frame[:, w_v // 2:, :]
    gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces_f = cascade_frontal.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3,
                                               minSize=(40, 40))
    faces_p = cascade_profile.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3,
                                               minSize=(40, 40))
    # Flipped (detect profile facing other way)
    gray_flip = cv2.flip(gray, 1)
    faces_pf = cascade_profile.detectMultiScale(gray_flip, scaleFactor=1.1, minNeighbors=3,
                                                minSize=(40, 40))

    tag = None
    faces = []
    if len(faces_f) > 0:
        tag = "frontal"
        faces = faces_f
    elif len(faces_p) > 0:
        tag = "profile"
        faces = faces_p
    elif len(faces_pf) > 0:
        tag = "profile_flipped"
        faces = faces_pf

    if tag:
        print(f"  Frame {fi:6d}: [{tag}] {len(faces)} face(s)")
        best_hits.append((fi, frame, faces, tag, right))
        if len(best_hits) >= 8:
            break

cap.release()

if not best_hits:
    print("No face detected by Haarcascade either.")
    # Save a frame for manual inspection at full resolution
    cap2 = cv2.VideoCapture(VIDEO_PATH)
    cap2.set(cv2.CAP_PROP_POS_FRAMES, int(total * 0.1))
    ret, frame = cap2.read()
    cap2.release()
    cv2.imwrite("full_res_frame.png", frame)
    right = frame[:, w_v // 2:, :]
    cv2.imwrite("right_half_frame.png", right)
    print("Saved full_res_frame.png and right_half_frame.png for inspection")
else:
    for rank, (fi, frame, faces, tag, right_half) in enumerate(best_hits[:5]):
        annotated = frame.copy()
        for (x, y, fw, fh) in faces:
            x_full = x + w_v // 2
            cv2.rectangle(annotated, (x_full, y), (x_full + fw, y + fh), (0, 255, 0), 2)
            cv2.putText(annotated, tag, (x_full, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        out = f"hit_{rank+1}_frame{fi}_{tag}.png"
        cv2.imwrite(out, annotated)
        print(f"Saved {out}")
