import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import os

# ── CONFIG ────────────────────────────────────────────────
VIDEO_PATH  = "check_thermal.mp4"      
OUTPUT_CSV  = "nose_tracking.csv"
OUTPUT_DIR  = "tracking_frames"        # Folder to save visualized frames
SAMPLE_EVERY_N_FRAMES = 25             
PATCH_RADIUS = 2                       

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ── 9 nose landmark positions (ratio of face bounding box) ─
NOSE_REGIONS = {
    "nose_bridge": (0.3732, 0.4859),
    "nose_tip": (0.4155, 0.4789),
    "left_nostril": (0.4577, 0.4789),
    "right_nostril": (0.507, 0.4859),
    "nose_left_ala": (0.5423, 0.493),
    "nose_right_ala": (0.3873, 0.4437),
    "philtrum": (0.4155, 0.3944),
    "nose_base_left": (0.4577, 0.3873),
    "nose_base_right": (0.493, 0.4225),
}

def thermal_intensity(patch_bgr: np.ndarray) -> float:
    b = patch_bgr[:, :, 0].astype(float)
    g = patch_bgr[:, :, 1].astype(float)
    r = patch_bgr[:, :, 2].astype(float)
    return float((r / 2 + g / 4 + b / 8).mean())

def run_tracker():
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {VIDEO_PATH}")

    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {total_frames} frames  |  {fps} fps")

    records   = []
    last_face = None 

    for frame_idx in range(0, total_frames, SAMPLE_EVERY_N_FRAMES):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        # Create a copy for visualization so we don't analyze the drawn-on pixels
        vis_frame = frame.copy()
        
        timestamp = round(frame_idx / fps, 3)
        H, W = frame.shape[:2]

        r_chan = frame[:, :, 2]
        faces  = face_cascade.detectMultiScale(
            r_chan, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60)
        )

        row = {
            "timestamp_s":  timestamp,
            "frame_idx":    frame_idx,
            "face_detected": len(faces) > 0,
        }

        if len(faces) > 0:
            # Pick the largest face
            x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
            last_face = (x, y, w, h)
        
        if last_face is not None:
            x, y, w, h = last_face
            row.update({"face_x": x, "face_y": y, "face_w": w, "face_h": h})
            
            # Draw Face Bounding Box (Cyan)
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

            for name, (rx, ry) in NOSE_REGIONS.items():
                px = int(x + rx * w)
                py = int(y + ry * h)
                px = max(PATCH_RADIUS, min(W - PATCH_RADIUS - 1, px))
                py = max(PATCH_RADIUS, min(H - PATCH_RADIUS - 1, py))

                patch = frame[
                    py - PATCH_RADIUS : py + PATCH_RADIUS + 1,
                    px - PATCH_RADIUS : px + PATCH_RADIUS + 1,
                ]

                row[f"{name}_px"] = px
                row[f"{name}_py"] = py
                row[f"{name}_TI"] = round(thermal_intensity(patch), 2)

                # Draw Landmark (Green circle)
                cv2.circle(vis_frame, (px, py), 2, (0, 255, 0), -1)
            
            # Add timestamp text to image
            cv2.putText(vis_frame, f"Time: {timestamp}s", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Save the visualized frame
            img_path = os.path.join(OUTPUT_DIR, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(img_path, vis_frame)

        records.append(row)

    cap.release()

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDone! CSV saved to {OUTPUT_CSV}")
    print(f"Visualized frames saved to folder: {OUTPUT_DIR}")

    return df

if __name__ == "__main__":
    run_tracker()