"""
Thermal nostril keypoint detector — training pipeline for Google Colab.

Usage on Colab:
  1. Mount Drive:   from google.colab import drive; drive.mount('/content/drive')
  2. Unzip data:    !unzip /content/drive/MyDrive/nostril_colab_data.zip -d /content/nostril_data
  3. Run:           !python 12_train_nostril_model.py

Outputs (saved to /content/drive/MyDrive/):
  nostril_model.pt          trained weights
  nostril_predictions.csv   per-frame predictions (run on all labeled frames)
  training_curves.png       loss / MAE curves
"""

# ── 0. COLAB SETUP ─────────────────────────────────────────────────────────────
import os, sys

# Paths — adjust if you mounted Drive differently
DATA_DIR   = "/content/nostril_data"
FRAMES_DIR = os.path.join(DATA_DIR, "frames")
LABELS_CSV = os.path.join(DATA_DIR, "nostril_labels.csv")
SAVE_DIR   = "/content/drive/MyDrive"     # where to save outputs

os.makedirs(SAVE_DIR, exist_ok=True)

# ── 1. IMPORTS ─────────────────────────────────────────────────────────────────
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.transforms import functional as TF
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import random, copy, time

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ── 2. CONFIG ──────────────────────────────────────────────────────────────────

# Nose zoom panel dimensions (780x540 — right half of composite frame)
PANEL_W, PANEL_H = 780, 540

# Tight crop around the nostril zone (where nostrils actually appear in panel space)
# Derived from label statistics: panel_x ~ 300-620, panel_y ~ 0-220
CROP_X1, CROP_X2 = 250, 650    # 400 px wide
CROP_Y1, CROP_Y2 = 0,   220    # 220 px tall
CROP_W = CROP_X2 - CROP_X1     # 400
CROP_H = CROP_Y2 - CROP_Y1     # 220

IMG_SIZE    = 224               # resize crop to this for backbone
BATCH_SIZE  = 16
EPOCHS      = 80
LR          = 3e-4
WEIGHT_DECAY= 1e-4
N_FOLDS     = 5

# Augmentation strengths
AUG_BRIGHTNESS = 0.35
AUG_CONTRAST   = 0.30
AUG_SHIFT_PX   = 25            # max random shift in panel pixels
AUG_SCALE      = 0.12          # max random zoom ± fraction
AUG_HFLIP_PROB = 0.5

# Coordinate mapping helpers
NOSE_ORIG_X = 260
NOSE_ORIG_Y = 145
ZOOM_FACTOR = 6

def panel_to_orig(px, py):
    return px / ZOOM_FACTOR + NOSE_ORIG_X, py / ZOOM_FACTOR + NOSE_ORIG_Y

# ── 3. DATASET ─────────────────────────────────────────────────────────────────

class NostrilDataset(Dataset):
    """
    Each sample:
      image  : (3, IMG_SIZE, IMG_SIZE)  float32 tensor, ImageNet-normalised
      target : (4,)  float32 tensor  [lx, ly, rx, ry] normalised to [0,1]
               within the CROP region
    """

    def __init__(self, df, augment=True):
        self.df      = df.reset_index(drop=True)
        self.augment = augment
        self.norm    = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225],
        )

    def __len__(self):
        return len(self.df)

    def _load_panel(self, frame_file):
        path = os.path.join(FRAMES_DIR, frame_file)
        img  = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # H=540, W=780, C=3

    def _labels_to_crop_norm(self, lx_p, ly_p, rx_p, ry_p):
        """Panel coords → normalised within crop region [0,1]."""
        lx = (lx_p - CROP_X1) / CROP_W
        ly = (ly_p - CROP_Y1) / CROP_H
        rx = (rx_p - CROP_X1) / CROP_W
        ry = (ry_p - CROP_Y1) / CROP_H
        return np.array([lx, ly, rx, ry], dtype=np.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        panel = self._load_panel(row["frame_file"])   # H×W×3 uint8

        # Panel-space label coords
        lx_p = float(row["left_panel_x"])
        ly_p = float(row["left_panel_y"])
        rx_p = float(row["right_panel_x"])
        ry_p = float(row["right_panel_y"])

        if self.augment:
            panel, lx_p, ly_p, rx_p, ry_p = _augment(
                panel, lx_p, ly_p, rx_p, ry_p)

        # Crop to nostril zone
        crop = panel[CROP_Y1:CROP_Y2, CROP_X1:CROP_X2]  # 220×400×3

        # Resize to IMG_SIZE×IMG_SIZE
        crop_rs = cv2.resize(crop, (IMG_SIZE, IMG_SIZE),
                             interpolation=cv2.INTER_LINEAR)

        # Adjust labels for crop
        lx_c = lx_p - CROP_X1;  rx_c = rx_p - CROP_X1
        ly_c = ly_p - CROP_Y1;  ry_c = ry_p - CROP_Y1

        # Scale labels for resize (CROP_W→IMG_SIZE, CROP_H→IMG_SIZE)
        lx_n = lx_c / CROP_W;   rx_n = rx_c / CROP_W
        ly_n = ly_c / CROP_H;   ry_n = ry_c / CROP_H

        target = np.array([lx_n, ly_n, rx_n, ry_n], dtype=np.float32)
        target = np.clip(target, 0.0, 1.0)

        # To tensor
        img_t = torch.from_numpy(crop_rs.transpose(2, 0, 1)).float() / 255.0
        img_t = self.norm(img_t)

        return img_t, torch.from_numpy(target)

    @staticmethod
    def decode_target(target_norm):
        """
        Inverse of normalisation: returns panel coords (float).
        target_norm: tensor/array (4,) in [0,1]
        """
        t = np.asarray(target_norm)
        lx_p = t[0] * CROP_W + CROP_X1
        ly_p = t[1] * CROP_H + CROP_Y1
        rx_p = t[2] * CROP_W + CROP_X1
        ry_p = t[3] * CROP_H + CROP_Y1
        return lx_p, ly_p, rx_p, ry_p


# ── 4. AUGMENTATION ────────────────────────────────────────────────────────────

def _augment(panel, lx, ly, rx, ry):
    """
    Apply random augmentations to panel (H×W×3 uint8) and update coords.
    Returns augmented panel and updated (lx, ly, rx, ry) in panel space.
    """
    h, w = panel.shape[:2]

    # Random horizontal flip — swap left/right labels
    if random.random() < AUG_HFLIP_PROB:
        panel = panel[:, ::-1, :].copy()
        lx, rx = w - 1 - rx, w - 1 - lx   # swap and mirror
        # also swap ly/ry (usually same, but keep consistent)
        ly, ry = ry, ly

    # Random brightness / contrast
    panel = panel.astype(np.float32)
    brt = random.uniform(-AUG_BRIGHTNESS, AUG_BRIGHTNESS) * 255
    ctr = random.uniform(1 - AUG_CONTRAST, 1 + AUG_CONTRAST)
    panel = panel * ctr + brt
    panel = np.clip(panel, 0, 255).astype(np.uint8)

    # Random spatial shift (translate)
    dx = random.randint(-AUG_SHIFT_PX, AUG_SHIFT_PX)
    dy = random.randint(-AUG_SHIFT_PX, AUG_SHIFT_PX)
    M  = np.float32([[1, 0, dx], [0, 1, dy]])
    panel = cv2.warpAffine(panel, M, (w, h),
                           borderMode=cv2.BORDER_REFLECT_101)
    lx += dx;  rx += dx
    ly += dy;  ry += dy

    # Random scale (zoom)
    scale  = random.uniform(1 - AUG_SCALE, 1 + AUG_SCALE)
    cx, cy = w / 2, h / 2
    M2 = cv2.getRotationMatrix2D((cx, cy), 0, scale)
    panel = cv2.warpAffine(panel, M2, (w, h),
                           borderMode=cv2.BORDER_REFLECT_101)
    # Apply same transform to coords
    pts   = np.float32([[lx, ly], [rx, ry]]).reshape(-1, 1, 2)
    pts_t = cv2.transform(pts, M2).reshape(-1, 2)
    lx, ly = pts_t[0]; rx, ry = pts_t[1]

    return panel, float(lx), float(ly), float(rx), float(ry)


# ── 5. MODEL ───────────────────────────────────────────────────────────────────

class NostrilNet(nn.Module):
    """
    MobileNetV3-Small backbone (lightweight, good for keypoints) +
    regression head predicting 4 normalised coords.
    """

    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v3_small(weights=weights)

        # Use all but the final classifier
        self.features = backbone.features
        self.avgpool  = backbone.avgpool

        # Custom regression head
        in_features = 576   # MobileNetV3-Small avgpool output
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 256),
            nn.Hardswish(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.Hardswish(),
            nn.Linear(64, 4),
            nn.Sigmoid(),          # output in [0,1]
        )

        # Freeze backbone initially; unfreeze after warm-up
        self._set_backbone_grad(False)

    def _set_backbone_grad(self, requires_grad):
        for p in self.features.parameters():
            p.requires_grad = requires_grad

    def unfreeze_backbone(self):
        self._set_backbone_grad(True)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return self.head(x)


# ── 6. WING LOSS ───────────────────────────────────────────────────────────────

class WingLoss(nn.Module):
    """
    Wing Loss (Feng et al. 2018) — designed for facial landmark regression.
    More sensitive to small errors than L2.
    """
    def __init__(self, w=10.0, epsilon=2.0):
        super().__init__()
        self.w = w
        self.e = epsilon
        self.C = w - w * np.log(1 + w / epsilon)

    def forward(self, pred, target):
        diff = torch.abs(pred - target)
        loss = torch.where(
            diff < self.w,
            self.w * torch.log(1 + diff / self.e),
            diff - self.C,
        )
        return loss.mean()


# ── 7. TRAINING UTILITIES ──────────────────────────────────────────────────────

def pixel_mae(pred, target):
    """Mean absolute error in panel pixels (un-normalise first)."""
    p = pred.detach().cpu().numpy()
    t = target.detach().cpu().numpy()
    # Un-normalise
    p[:, [0, 2]] *= CROP_W;  p[:, [1, 3]] *= CROP_H
    t[:, [0, 2]] *= CROP_W;  t[:, [1, 3]] *= CROP_H
    # MAE in pixels
    return float(np.abs(p - t).mean())


def run_epoch(model, loader, criterion, optimizer, train):
    model.train(train)
    total_loss = total_mae = n = 0
    with torch.set_grad_enabled(train):
        for imgs, targets in loader:
            imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
            preds = model(imgs)
            loss  = criterion(preds, targets)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(imgs)
            total_mae  += pixel_mae(preds, targets) * len(imgs)
            n          += len(imgs)
    return total_loss / n, total_mae / n


def train_fold(train_df, val_df, fold_id):
    print(f"\n  Fold {fold_id}  train={len(train_df)}  val={len(val_df)}")

    train_ds = NostrilDataset(train_df, augment=True)
    val_ds   = NostrilDataset(val_df,   augment=False)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=2, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=2, pin_memory=True)

    model     = NostrilNet(pretrained=True).to(DEVICE)
    criterion = WingLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_mae  = float("inf")
    best_weights  = None
    train_losses  = []
    val_maes      = []
    unfreeze_done = False

    for epoch in range(1, EPOCHS + 1):
        # Unfreeze backbone after warm-up
        if epoch == 20 and not unfreeze_done:
            model.unfreeze_backbone()
            optimizer.add_param_group(
                {"params": model.features.parameters(), "lr": LR * 0.1}
            )
            unfreeze_done = True
            print("    Backbone unfrozen.")

        tr_loss, _       = run_epoch(model, train_dl, criterion, optimizer, train=True)
        val_loss, val_mae = run_epoch(model, val_dl, criterion, optimizer, train=False)
        scheduler.step()

        train_losses.append(tr_loss)
        val_maes.append(val_mae)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_weights = copy.deepcopy(model.state_dict())

        if epoch % 10 == 0:
            print(f"    Ep {epoch:3d}  tr_loss={tr_loss:.4f}  "
                  f"val_mae={val_mae:.2f}px  best={best_val_mae:.2f}px")

    model.load_state_dict(best_weights)
    print(f"  Best val MAE: {best_val_mae:.2f} px")
    return model, best_val_mae, train_losses, val_maes


# ── 8. CROSS-VALIDATION + FINAL TRAIN ─────────────────────────────────────────

def main():
    df   = pd.read_csv(LABELS_CSV)
    good = df[df["skipped"] == False].reset_index(drop=True)
    print(f"Loaded {len(good)} labeled frames from {LABELS_CSV}")

    # Verify all frames exist
    missing = [r for _, r in good.iterrows()
               if not os.path.exists(os.path.join(FRAMES_DIR, r["frame_file"]))]
    if missing:
        print(f"WARNING: {len(missing)} frame files not found — dropping.")
        good = good[~good["frame_file"].isin([r["frame_file"] for r in missing])]

    print(f"Using {len(good)} samples for training\n")

    # ── Cross-validation ─────────────────────────────────────────────────────
    kf      = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_maes     = []
    all_losses    = []
    all_val_maes  = []

    for fold_id, (tr_idx, va_idx) in enumerate(kf.split(good), 1):
        model, best_mae, tr_l, va_m = train_fold(
            good.iloc[tr_idx], good.iloc[va_idx], fold_id)
        fold_maes.append(best_mae)
        all_losses.append(tr_l)
        all_val_maes.append(va_m)

    print(f"\nCross-validation MAE: {np.mean(fold_maes):.2f} ± {np.std(fold_maes):.2f} px")

    # ── Final model on all data ───────────────────────────────────────────────
    print("\nTraining final model on all data ...")
    final_model, _, final_losses, _ = train_fold(good, good.iloc[:1], fold_id=0)
    # (val on 1 sample is just for API compatibility; we care about train convergence)

    # Save model
    model_path = os.path.join(SAVE_DIR, "nostril_model.pt")
    torch.save({
        "model_state":  final_model.state_dict(),
        "crop_x1": CROP_X1, "crop_x2": CROP_X2,
        "crop_y1": CROP_Y1, "crop_y2": CROP_Y2,
        "img_size": IMG_SIZE,
        "nose_orig_x": NOSE_ORIG_X, "nose_orig_y": NOSE_ORIG_Y,
        "zoom_factor": ZOOM_FACTOR,
        "cv_mae_mean": float(np.mean(fold_maes)),
        "cv_mae_std":  float(np.std(fold_maes)),
    }, model_path)
    print(f"Model saved: {model_path}")

    # ── Training curves ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for i, (tr_l, va_m) in enumerate(zip(all_losses, all_val_maes)):
        axes[0].plot(tr_l,  alpha=0.6, label=f"fold {i+1}")
        axes[1].plot(va_m,  alpha=0.6, label=f"fold {i+1}")
    axes[0].set_title("Training loss (Wing)"); axes[0].set_xlabel("Epoch")
    axes[1].set_title("Val MAE (px)");         axes[1].set_xlabel("Epoch")
    for ax in axes:
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
    axes[1].axhline(np.mean(fold_maes), color="red", linestyle="--",
                    label=f"mean best = {np.mean(fold_maes):.1f}px")
    axes[1].legend(fontsize=7)
    plt.suptitle(f"Nostril Detector — 5-fold CV  "
                 f"MAE = {np.mean(fold_maes):.1f} ± {np.std(fold_maes):.1f} px",
                 fontweight="bold")
    plt.tight_layout()
    curve_path = os.path.join(SAVE_DIR, "training_curves.png")
    plt.savefig(curve_path, dpi=120)
    print(f"Curves saved: {curve_path}")

    # ── Predictions on all labeled frames ─────────────────────────────────────
    final_model.eval()
    norm = transforms.Normalize(mean=[0.485,0.456,0.406],
                                std =[0.229,0.224,0.225])
    pred_rows = []
    for _, row in good.iterrows():
        path  = os.path.join(FRAMES_DIR, row["frame_file"])
        panel = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        crop  = panel[CROP_Y1:CROP_Y2, CROP_X1:CROP_X2]
        crop_rs = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
        img_t  = norm(torch.from_numpy(
            crop_rs.transpose(2,0,1)).float() / 255.0).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred = final_model(img_t).cpu().numpy()[0]

        lx_p = pred[0]*CROP_W + CROP_X1
        ly_p = pred[1]*CROP_H + CROP_Y1
        rx_p = pred[2]*CROP_W + CROP_X1
        ry_p = pred[3]*CROP_H + CROP_Y1

        pred_rows.append({
            "frame_idx":       row["frame_idx"],
            "frame_file":      row["frame_file"],
            # Predicted panel coords
            "pred_left_panel_x":  round(lx_p, 2),
            "pred_left_panel_y":  round(ly_p, 2),
            "pred_right_panel_x": round(rx_p, 2),
            "pred_right_panel_y": round(ry_p, 2),
            # Predicted original thermal coords
            "pred_left_orig_x":  round(lx_p/ZOOM_FACTOR + NOSE_ORIG_X, 2),
            "pred_left_orig_y":  round(ly_p/ZOOM_FACTOR + NOSE_ORIG_Y, 2),
            "pred_right_orig_x": round(rx_p/ZOOM_FACTOR + NOSE_ORIG_X, 2),
            "pred_right_orig_y": round(ry_p/ZOOM_FACTOR + NOSE_ORIG_Y, 2),
            # Manual labels for comparison
            "manual_left_orig_x":  row["left_orig_x"],
            "manual_left_orig_y":  row["left_orig_y"],
            "manual_right_orig_x": row["right_orig_x"],
            "manual_right_orig_y": row["right_orig_y"],
        })

    pred_df = pd.DataFrame(pred_rows)
    # Pixel error
    pred_df["err_left"]  = np.sqrt(
        (pred_df.pred_left_orig_x  - pred_df.manual_left_orig_x)**2 +
        (pred_df.pred_left_orig_y  - pred_df.manual_left_orig_y)**2)
    pred_df["err_right"] = np.sqrt(
        (pred_df.pred_right_orig_x - pred_df.manual_right_orig_x)**2 +
        (pred_df.pred_right_orig_y - pred_df.manual_right_orig_y)**2)

    pred_path = os.path.join(SAVE_DIR, "nostril_predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"Predictions saved: {pred_path}")
    print(f"\nTrain-set MAE  left={pred_df.err_left.mean():.2f}px  "
          f"right={pred_df.err_right.mean():.2f}px")
    print(f"\nDone. Check {SAVE_DIR}/ for model, curves, predictions.")


if __name__ == "__main__":
    main()
