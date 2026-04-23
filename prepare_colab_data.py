"""
Run this LOCALLY to package the labeled thermal frames for Colab.

Creates:  nostril_colab_data.zip
  nostril_labels.csv
  frames/frame_XXXXX.jpg   (89 labeled frames only, nose-panel crops)

Upload nostril_colab_data.zip to Google Drive, then run 12_train_nostril_model.py on Colab.
"""

import cv2, numpy as np, pandas as pd, zipfile, io
from pathlib import Path

FRAMES_DIR   = Path("thermal_face_frames")
LABELS_CSV   = "nostril_labels.csv"
OUT_ZIP      = "nostril_colab_data.zip"
FACE_PANEL_W = 675    # nose zoom panel starts here

df = pd.read_csv(LABELS_CSV)
good = df[df["skipped"] == False].reset_index(drop=True)
print(f"Packing {len(good)} labeled frames ...")

with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
    # CSV
    zf.writestr("nostril_labels.csv", df.to_csv(index=False))

    # Frames — save only the nose zoom panel (right half of composite)
    for _, row in good.iterrows():
        fp = FRAMES_DIR / row["frame_file"]
        img = cv2.imread(str(fp))
        if img is None:
            print(f"  MISSING: {fp.name}")
            continue
        nose_panel = img[:, FACE_PANEL_W:, :]   # 780x540
        ok, buf = cv2.imencode(".jpg", nose_panel, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if ok:
            zf.writestr(f"frames/{row['frame_file']}", buf.tobytes())

print(f"Done. Saved: {OUT_ZIP}  ({Path(OUT_ZIP).stat().st_size/1e6:.1f} MB)")
print("Upload this zip to Google Drive and mount in Colab.")
