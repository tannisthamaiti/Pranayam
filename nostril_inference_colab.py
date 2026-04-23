# ============================================================
# Nostril Detector — Colab Inference
# ============================================================
# HOW TO USE:
#   1. Upload nostril_model.pt to Colab (or mount Drive)
#   2. Upload your thermal frames to /content/frames/
#      OR mount Drive and set FRAMES_DIR below
#   3. Runtime → Run all
# ============================================================


# ── Cell 1 · Install (run once) ──────────────────────────────
# !pip install torch torchvision --quiet   # already available on Colab GPU runtimes


# ── Cell 2 · Imports ─────────────────────────────────────────
import os, glob, math
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
import matplotlib.pyplot as plt
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)


# ── Cell 3 · Config  ─────────────────────────────────────────
# ↓↓↓ Only edit these two lines ↓↓↓
CHECKPOINT_PATH = "/content/nostril_model.pt"   # path to your .pt file
FRAMES_DIR      = "/content/frames"             # folder with thermal frame images
# ↑↑↑ ─────────────────────────────────────────────────────────

# These must match your training config (do not change)
CROP_X1, CROP_X2 = 250, 650
CROP_Y1, CROP_Y2 = 0,   220
CROP_W   = CROP_X2 - CROP_X1   # 400
CROP_H   = CROP_Y2 - CROP_Y1   # 220
IMG_SIZE = 224
ZOOM_FACTOR  = 6
NOSE_ORIG_X  = 260
NOSE_ORIG_Y  = 145

# Raw thermal frame (640x480) nose region — used when input is raw frames
RAW_NOSE_X1, RAW_NOSE_X2 = 260, 390   # match extract_thermal_frames.py
RAW_NOSE_Y1, RAW_NOSE_Y2 = 145, 235
RAW_NOSE_W = RAW_NOSE_X2 - RAW_NOSE_X1   # 130
RAW_NOSE_H = RAW_NOSE_Y2 - RAW_NOSE_Y1   # 90
# After 6x upscale this becomes the 780x540 nose panel the model expects


# ── Cell 4 · Model definition (must match training) ──────────
class NostrilNet(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.mobilenet_v3_small(weights=None)
        self.features = backbone.features
        self.avgpool  = backbone.avgpool
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(576, 256),
            nn.Hardswish(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.Hardswish(),
            nn.Linear(64, 4),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.head(self.avgpool(self.features(x)))


# ── Cell 5 · Load checkpoint ─────────────────────────────────
ckpt  = torch.load(CHECKPOINT_PATH, map_location="cpu")
model = NostrilNet()
model.load_state_dict(ckpt["model_state"])
model.to(DEVICE).eval()

print(f"Checkpoint loaded.")
print(f"  Trained CV MAE: {ckpt['cv_mae_mean']:.2f} ± {ckpt['cv_mae_std']:.2f} px")
print(f"  Crop region   : x=[{ckpt['crop_x1']},{ckpt['crop_x2']}]  "
      f"y=[{ckpt['crop_y1']},{ckpt['crop_y2']}]")


# ── Cell 6 · Preprocessing ───────────────────────────────────
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std =[0.229, 0.224, 0.225],
)

def preprocess(frame_bgr):
    """
    frame_bgr : numpy array (H, W, 3) in BGR from cv2.imread.
                Accepts either:
                  (a) raw 640x480 thermal frame  — automatically extracts+upscales nose panel
                  (b) 780x540 nose zoom panel    — used directly (same format as training)
    Returns   : tensor (1, 3, 224, 224) ready for the model.
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    h, w = rgb.shape[:2]
    if w <= 640:
        # Raw thermal frame: extract nose region and upscale to match training format
        nose_raw   = rgb[RAW_NOSE_Y1:RAW_NOSE_Y2, RAW_NOSE_X1:RAW_NOSE_X2]
        nose_panel = cv2.resize(nose_raw,
                                (RAW_NOSE_W * ZOOM_FACTOR, RAW_NOSE_H * ZOOM_FACTOR),
                                interpolation=cv2.INTER_CUBIC)   # → 780×540
        rgb = nose_panel
    # else already a 780×540 nose panel — use as-is

    crop = rgb[CROP_Y1:CROP_Y2, CROP_X1:CROP_X2]           # 220 × 400
    rs   = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))            # 224 × 224
    t    = torch.from_numpy(rs.transpose(2, 0, 1)).float() / 255.0
    return normalize(t).unsqueeze(0)                         # (1, 3, 224, 224)


# ── Cell 7 · Inference helpers ───────────────────────────────
def predict(frame_bgr):
    """
    Returns dict with panel-space and original-thermal-space coordinates.
    All values are in pixels (float).
    """
    inp = preprocess(frame_bgr).to(DEVICE)
    with torch.no_grad():
        out = model(inp).cpu().numpy()[0]    # (4,) normalised [0,1]

    # Un-normalise to panel pixel coords
    lx_p = out[0] * CROP_W + CROP_X1
    ly_p = out[1] * CROP_H + CROP_Y1
    rx_p = out[2] * CROP_W + CROP_X1
    ry_p = out[3] * CROP_H + CROP_Y1

    # Convert panel → original thermal frame coords
    lx_o = lx_p / ZOOM_FACTOR + NOSE_ORIG_X
    ly_o = ly_p / ZOOM_FACTOR + NOSE_ORIG_Y
    rx_o = rx_p / ZOOM_FACTOR + NOSE_ORIG_X
    ry_o = ry_p / ZOOM_FACTOR + NOSE_ORIG_Y

    return {
        "panel": {"left": (lx_p, ly_p), "right": (rx_p, ry_p)},
        "orig":  {"left": (lx_o, ly_o), "right": (rx_o, ry_o)},
    }


def visualise(frame_bgr, preds, ax=None):
    """
    Draws the detected nostril points on the crop region.
    """
    rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    crop = rgb[CROP_Y1:CROP_Y2, CROP_X1:CROP_X2].copy()

    # Panel coords shifted to crop-local space for display
    lx_c = preds["panel"]["left"][0]  - CROP_X1
    ly_c = preds["panel"]["left"][1]  - CROP_Y1
    rx_c = preds["panel"]["right"][0] - CROP_X1
    ry_c = preds["panel"]["right"][1] - CROP_Y1

    show_alone = ax is None
    if show_alone:
        fig, ax = plt.subplots(figsize=(7, 3))

    ax.imshow(crop)
    ax.plot(lx_c, ly_c, "o", color="#00FF7F", markersize=9,
            markeredgecolor="white", markeredgewidth=1.5, label="Left nostril")
    ax.plot(rx_c, ry_c, "o", color="#FF4500", markersize=9,
            markeredgecolor="white", markeredgewidth=1.5, label="Right nostril")
    ax.legend(loc="upper right", fontsize=8)
    ax.axis("off")

    if show_alone:
        plt.tight_layout()
        plt.show()


# ── Cell 8 · Run on all frames in FRAMES_DIR ─────────────────
image_paths = sorted(
    glob.glob(os.path.join(FRAMES_DIR, "*.jpg")) +
    glob.glob(os.path.join(FRAMES_DIR, "*.jpeg")) +
    glob.glob(os.path.join(FRAMES_DIR, "*.png"))
)

if not image_paths:
    print(f"No images found in {FRAMES_DIR}")
else:
    print(f"Found {len(image_paths)} frames — running inference ...")

    results = []
    ncols   = min(4, len(image_paths))
    nrows   = math.ceil(len(image_paths) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows))
    axes_flat = np.array(axes).flatten() if len(image_paths) > 1 else [axes]

    for i, path in enumerate(image_paths):
        frame = cv2.imread(path)
        if frame is None:
            print(f"  Cannot read: {path}")
            axes_flat[i].axis("off")
            continue

        preds = predict(frame)
        visualise(frame, preds, ax=axes_flat[i])
        axes_flat[i].set_title(os.path.basename(path), fontsize=8)

        lx_o, ly_o = preds["orig"]["left"]
        rx_o, ry_o = preds["orig"]["right"]
        results.append({
            "frame_file":       os.path.basename(path),
            "pred_left_orig_x":  round(lx_o, 2),
            "pred_left_orig_y":  round(ly_o, 2),
            "pred_right_orig_x": round(rx_o, 2),
            "pred_right_orig_y": round(ry_o, 2),
        })
        print(f"  {os.path.basename(path):35s}  "
              f"L=({lx_o:.1f},{ly_o:.1f})  R=({rx_o:.1f},{ry_o:.1f})")

    for j in range(len(image_paths), len(axes_flat)):
        axes_flat[j].axis("off")

    plt.suptitle("Nostril Detections", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("/content/nostril_detections.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nVisualization saved → /content/nostril_detections.png")


# ── Cell 9 · Save predictions to CSV ─────────────────────────
import pandas as pd

if results:
    df_out = pd.DataFrame(results)
    df_out.to_csv("/content/nostril_predictions_inference.csv", index=False)
    print("Predictions saved → /content/nostril_predictions_inference.csv")
    display(df_out.head(10))
