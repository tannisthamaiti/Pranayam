"""
PPG Pre/Post Session — Unsupervised & Semi-Supervised Analysis
==============================================================
Subject: NEEMA  |  Device: Pulse oximeter  |  Sampling rate: ~60 Hz

Files expected (same directory as this script):
  NEEMA_D1_PRE_20251226122201.csv          — SPO2, PULSE (1 Hz)
  NEEMA_D1_PRE_20251226122201_wave.csv     — PPG waveform (60 Hz)
  NEEMA_D1_POST_20251226124134.csv         — SPO2, PULSE (1 Hz)
  NEEMA_D1_POST_20251226124134_wave.csv    — PPG waveform (60 Hz)

Analyses included
-----------------
[UNSUPERVISED]
  1. RR-interval anomaly detection       — Isolation Forest + LOF
  2. Poincaré DBSCAN clustering          — beat-class discovery
  3. PPG beat morphology clustering      — k-means on beat shapes
  4. Change-point detection              — PELT on RR series

[SEMI-SUPERVISED]
  5. Feature window embedding            — PCA / UMAP on HRV windows
  6. Label propagation                   — sparse manual labels → all windows
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

from scipy.signal import find_peaks
from scipy.interpolate import interp1d

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelSpreading

# Optional: UMAP (install with: pip install umap-learn)
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("umap-learn not installed — skipping UMAP, using PCA only.")

# ─── Optional: ruptures for change-point detection ─────────────────────────
try:
    import ruptures as rpt
    RUPTURES_AVAILABLE = True
except ImportError:
    RUPTURES_AVAILABLE = False
    print("ruptures not installed — using manual PELT fallback.")
    print("Install with: pip install ruptures")

# ═══════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

FS = 60  # PPG sampling rate (Hz)
DATA_DIR = "NEEMA_Day1-20260408T100839Z-3-001\\NEEMA_Day1\\PPG"  # change if files are elsewhere

def load_data(data_dir=DATA_DIR):
    pre_csv   = pd.read_csv(f"{data_dir}\\PRE\\NEEMA_D1_PRE_20251226122201.csv")
    post_csv  = pd.read_csv(f"{data_dir}\\POST\\NEEMA_D1_POST_20251226124134.csv")
    pre_wave  = pd.read_csv(f"{data_dir}\\PRE\\NEEMA_D1_PRE_20251226122201_wave.csv")["Wave"].values.astype(float)
    post_wave = pd.read_csv(f"{data_dir}\\POST\\NEEMA_D1_POST_20251226124134_wave.csv")["Wave"].values.astype(float)

    # Mask sensor initialization artifact (255 bpm at row 0 in POST)
    post_csv.loc[post_csv["PULSE"] > 200, "PULSE"] = np.nan

    return pre_csv, post_csv, pre_wave, post_wave

pre_csv, post_csv, pre_wave, post_wave = load_data()
print(f"PRE  — {len(pre_wave):,} PPG samples, {len(pre_csv)} SpO2/pulse rows")
print(f"POST — {len(post_wave):,} PPG samples, {len(post_csv)} SpO2/pulse rows")


# ═══════════════════════════════════════════════════════════════════════════
# 2. PEAK DETECTION & RR INTERVALS
# ═══════════════════════════════════════════════════════════════════════════

def detect_peaks_rr(wave, fs=FS):
    """Detect systolic peaks and compute RR intervals in ms."""
    peaks, _ = find_peaks(
        wave,
        distance=int(fs * 0.4),       # minimum 400 ms between peaks (max ~150 bpm)
        height=np.percentile(wave, 55) # must exceed 55th percentile
    )
    rr_ms = np.diff(peaks) / fs * 1000   # convert samples → ms
    return peaks, rr_ms

pre_peaks,  pre_rr  = detect_peaks_rr(pre_wave)
post_peaks, post_rr = detect_peaks_rr(post_wave)

# Filter physiologically plausible RR intervals (300–2000 ms)
def clean_rr(rr, lo=300, hi=2000):
    mask = (rr >= lo) & (rr <= hi)
    return rr[mask], mask

pre_rr_c,  pre_mask  = clean_rr(pre_rr)
post_rr_c, post_mask = clean_rr(post_rr)

print(f"\nRR intervals — PRE: {len(pre_rr_c)} clean beats | POST: {len(post_rr_c)} clean beats")


# ═══════════════════════════════════════════════════════════════════════════
# 3. HRV FEATURE EXTRACTION (sliding windows)
# ═══════════════════════════════════════════════════════════════════════════

def hrv_features_window(rr_series, window_beats=20, step=5):
    """
    Slide a window over the RR series and compute HRV features per window.
    Returns a DataFrame with one row per window.
    """
    rows = []
    for start in range(0, len(rr_series) - window_beats, step):
        seg = rr_series[start : start + window_beats]
        diff = np.diff(seg)
        rows.append({
            "start_beat"  : start,
            "mean_rr"     : seg.mean(),
            "std_rr"      : seg.std(),
            "mean_hr"     : 60_000 / seg.mean(),
            "rmssd"       : np.sqrt(np.mean(diff ** 2)),
            "sdnn"        : seg.std(),
            "pnn50"       : np.sum(np.abs(diff) > 50) / len(diff) * 100,
            "range_rr"    : seg.max() - seg.min(),
            "cv_rr"       : seg.std() / seg.mean() * 100,   # coefficient of variation
        })
    return pd.DataFrame(rows)

pre_feat  = hrv_features_window(pre_rr_c)
post_feat = hrv_features_window(post_rr_c)

pre_feat["session"]  = "PRE"
post_feat["session"] = "POST"
all_feat = pd.concat([pre_feat, post_feat], ignore_index=True)

FEATURE_COLS = ["mean_rr", "std_rr", "mean_hr", "rmssd", "sdnn",
                "pnn50", "range_rr", "cv_rr"]

print(f"\nHRV windows — PRE: {len(pre_feat)} | POST: {len(post_feat)}")


# ═══════════════════════════════════════════════════════════════════════════
# 4. UNSUPERVISED — ANOMALY DETECTION ON RR SERIES
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "═"*60)
print("UNSUPERVISED 1: RR Anomaly Detection (Isolation Forest + LOF)")
print("═"*60)

scaler_rr = StandardScaler()
X_feat = scaler_rr.fit_transform(all_feat[FEATURE_COLS])

# — Isolation Forest —
iso = IsolationForest(n_estimators=200, contamination=0.1, random_state=42)
all_feat["iso_pred"]  = iso.fit_predict(X_feat)
all_feat["iso_score"] = iso.decision_function(X_feat)  # lower = more anomalous

# — Local Outlier Factor —
lof = LocalOutlierFactor(n_neighbors=10, contamination=0.1)
all_feat["lof_pred"]  = lof.fit_predict(X_feat)
all_feat["lof_score"] = lof.negative_outlier_factor_

# Summary
for label, grp in all_feat.groupby("session"):
    n_iso = (grp["iso_pred"] == -1).sum()
    n_lof = (grp["lof_pred"] == -1).sum()
    print(f"  {label}: Isolation Forest anomalies = {n_iso}/{len(grp)}  |  LOF anomalies = {n_lof}/{len(grp)}")


# ═══════════════════════════════════════════════════════════════════════════
# 5. UNSUPERVISED — POINCARÉ DBSCAN CLUSTERING
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "═"*60)
print("UNSUPERVISED 2: Poincaré Plot DBSCAN Clustering")
print("═"*60)

def poincare_matrix(rr_clean):
    """Return (N-1, 2) array of [RR[n], RR[n+1]] pairs."""
    return np.column_stack([rr_clean[:-1], rr_clean[1:]])

pre_poi  = poincare_matrix(pre_rr_c)
post_poi = poincare_matrix(post_rr_c)
all_poi  = np.vstack([pre_poi, post_poi])
poi_labels_session = np.array(["PRE"]  * len(pre_poi) +
                               ["POST"] * len(post_poi))

scaler_poi = StandardScaler()
X_poi = scaler_poi.fit_transform(all_poi)

db = DBSCAN(eps=0.8, min_samples=5)  # eps in normalised units ≈ 60 ms raw
poi_clusters = db.fit_predict(X_poi)

n_clusters = len(set(poi_clusters)) - (1 if -1 in poi_clusters else 0)
n_noise    = (poi_clusters == -1).sum()
print(f"  DBSCAN clusters found: {n_clusters}  |  noise points: {n_noise}")
for c in sorted(set(poi_clusters)):
    mask = poi_clusters == c
    rr_mean = all_poi[mask, 0].mean()
    label = f"Cluster {c}" if c != -1 else "Noise"
    print(f"    {label}: {mask.sum()} beats, mean RR[n] = {rr_mean:.0f} ms")


# ═══════════════════════════════════════════════════════════════════════════
# 6. UNSUPERVISED — PPG BEAT MORPHOLOGY CLUSTERING
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "═"*60)
print("UNSUPERVISED 3: PPG Beat Morphology Clustering (k-means)")
print("═"*60)

BEAT_SAMPLES = 60   # resample each beat to 60 points

def extract_beat_templates(wave, peaks, n_samples=BEAT_SAMPLES):
    """
    Extract one beat per peak (from preceding trough to next trough),
    resample to fixed length, and normalise amplitude to [0, 1].
    """
    templates, indices = [], []
    troughs, _ = find_peaks(-wave, distance=int(FS * 0.3))

    for i, p in enumerate(peaks):
        left  = troughs[troughs < p]
        right = troughs[troughs > p]
        if len(left) == 0 or len(right) == 0:
            continue
        l, r = left[-1], right[0]
        beat = wave[l:r+1]
        if len(beat) < 5:
            continue
        x_old = np.linspace(0, 1, len(beat))
        x_new = np.linspace(0, 1, n_samples)
        beat_resampled = interp1d(x_old, beat, kind="linear")(x_new)
        # amplitude normalise
        rng = beat_resampled.max() - beat_resampled.min()
        if rng > 0:
            beat_resampled = (beat_resampled - beat_resampled.min()) / rng
        templates.append(beat_resampled)
        indices.append(i)
    return np.array(templates), np.array(indices)

pre_temps,  pre_idx  = extract_beat_templates(pre_wave,  pre_peaks)
post_temps, post_idx = extract_beat_templates(post_wave, post_peaks)
all_temps = np.vstack([pre_temps, post_temps])
temp_session = np.array(["PRE"]  * len(pre_temps) +
                         ["POST"] * len(post_temps))

print(f"  Beat templates extracted — PRE: {len(pre_temps)} | POST: {len(post_temps)}")

# PCA on beat templates before k-means for speed
pca_beat = PCA(n_components=10, random_state=42)
X_beat   = pca_beat.fit_transform(all_temps)

km = KMeans(n_clusters=3, n_init=20, random_state=42)
beat_clusters = km.fit_predict(X_beat)

print(f"  Explained variance (10 PCs): {pca_beat.explained_variance_ratio_.sum()*100:.1f}%")
for c in range(3):
    mask = beat_clusters == c
    pre_frac = (temp_session[mask] == "PRE").mean() * 100
    print(f"    Morphology cluster {c}: {mask.sum()} beats | PRE fraction: {pre_frac:.0f}%")


# ═══════════════════════════════════════════════════════════════════════════
# 7. UNSUPERVISED — CHANGE-POINT DETECTION ON RR SERIES
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "═"*60)
print("UNSUPERVISED 4: Change-Point Detection on RR Series")
print("═"*60)

def simple_changepoint(rr, n_bkps=3, window=15):
    """
    Simple sliding-window variance change detector (fallback if ruptures not installed).
    Returns approximate breakpoint indices.
    """
    scores = []
    for i in range(window, len(rr) - window):
        left  = rr[i-window:i].std()
        right = rr[i:i+window].std()
        scores.append(abs(right - left))
    scores = np.array(scores)
    # pick top n_bkps non-overlapping peaks
    bkps = []
    used = set()
    for idx in np.argsort(scores)[::-1]:
        if not any(abs(idx - u) < window for u in used):
            bkps.append(idx + window)
            used.add(idx)
        if len(bkps) == n_bkps:
            break
    return sorted(bkps)

for session_label, rr_arr in [("PRE", pre_rr_c), ("POST", post_rr_c)]:
    if RUPTURES_AVAILABLE:
        signal_2d = rr_arr.reshape(-1, 1)
        algo = rpt.Pelt(model="rbf").fit(signal_2d)
        bkps = algo.predict(pen=20)
        bkps = [b for b in bkps if b < len(rr_arr)]
    else:
        bkps = simple_changepoint(rr_arr, n_bkps=3)

    print(f"  {session_label} change-points at beats: {bkps}")
    for i, b in enumerate(bkps):
        rr_before = rr_arr[max(0, b-10):b]
        rr_after  = rr_arr[b:b+10]
        if len(rr_before) and len(rr_after):
            print(f"    Beat {b}: mean RR {rr_before.mean():.0f} ms → {rr_after.mean():.0f} ms")


# ═══════════════════════════════════════════════════════════════════════════
# 8. SEMI-SUPERVISED — FEATURE WINDOW EMBEDDING (PCA / UMAP)
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "═"*60)
print("SEMI-SUPERVISED 1: HRV Feature Embedding (PCA + optional UMAP)")
print("═"*60)

X_all     = all_feat[FEATURE_COLS].values
scaler_ss = StandardScaler()
X_scaled  = scaler_ss.fit_transform(X_all)

pca2 = PCA(n_components=2, random_state=42)
X_pca = pca2.fit_transform(X_scaled)
all_feat["pca1"] = X_pca[:, 0]
all_feat["pca2"] = X_pca[:, 1]

print(f"  PCA variance explained: PC1={pca2.explained_variance_ratio_[0]*100:.1f}%  "
      f"PC2={pca2.explained_variance_ratio_[1]*100:.1f}%")

if UMAP_AVAILABLE:
    reducer = umap.UMAP(n_components=2, n_neighbors=10, min_dist=0.1, random_state=42)
    X_umap  = reducer.fit_transform(X_scaled)
    all_feat["umap1"] = X_umap[:, 0]
    all_feat["umap2"] = X_umap[:, 1]
    print("  UMAP embedding complete.")


# ═══════════════════════════════════════════════════════════════════════════
# 9. SEMI-SUPERVISED — LABEL PROPAGATION
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "═"*60)
print("SEMI-SUPERVISED 2: Label Propagation (sparse manual labels)")
print("═"*60)

"""
Strategy:
  - -1  = unlabelled (most windows)
  -  0  = "pre-rest"      (manually anchored to first few PRE windows)
  -  1  = "post-recovery" (manually anchored to first few POST windows)
  -  2  = "high-variability" (manually anchored to anomalous POST windows)
"""

n_pre  = len(pre_feat)
n_post = len(post_feat)
labels_manual = np.full(len(all_feat), -1, dtype=int)

# Anchor labels: first 3 PRE windows → class 0 (pre-rest)
labels_manual[:3] = 0

# Anchor labels: first 3 POST windows → class 1 (post-recovery)
labels_manual[n_pre : n_pre + 3] = 1

# Anchor labels: top-3 highest RMSSD POST windows → class 2 (high-variability)
rmssd_post_idx = (all_feat[all_feat["session"] == "POST"]["rmssd"]
                  .nlargest(3).index)
for idx in rmssd_post_idx:
    labels_manual[idx] = 2

n_labelled = (labels_manual != -1).sum()
print(f"  Labelled: {n_labelled} / {len(labels_manual)} windows ({n_labelled/len(labels_manual)*100:.0f}%)")

lp = LabelSpreading(kernel="knn", n_neighbors=7, max_iter=1000, alpha=0.2)
lp.fit(X_scaled, labels_manual)
all_feat["label_propagated"] = lp.predict(X_scaled)
all_feat["label_confidence"]  = lp.label_distributions_.max(axis=1)

CLASS_NAMES = {0: "pre-rest", 1: "post-recovery", 2: "high-variability"}
for c, name in CLASS_NAMES.items():
    grp = all_feat[all_feat["label_propagated"] == c]
    pre_n  = (grp["session"] == "PRE").sum()
    post_n = (grp["session"] == "POST").sum()
    conf   = grp["label_confidence"].mean()
    print(f"  {name}: {len(grp)} windows (PRE={pre_n}, POST={post_n}) | mean confidence={conf:.2f}")


# ═══════════════════════════════════════════════════════════════════════════
# 10. VISUALISATION  (saves to ppg_analysis_results.png)
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "═"*60)
print("Generating figure: ppg_analysis_results.png")
print("═"*60)

COLORS = {"PRE": "#3B8BD4", "POST": "#1D9E75"}
CLUSTER_COLORS = ["#3B8BD4", "#E8593C", "#9B59B6", "#E67E22", "#95A5A6"]

fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor("white")
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── (a) RR interval time series + anomalies ──────────────────────────────
ax1 = fig.add_subplot(gs[0, :2])
for session, rr_arr, color in [("PRE", pre_rr_c, COLORS["PRE"]),
                                 ("POST", post_rr_c, COLORS["POST"])]:
    ax1.plot(rr_arr, color=color, lw=1, alpha=0.8, label=session)
    # mark anomalous windows
    feat_s = all_feat[all_feat["session"] == session]
    anom   = feat_s[feat_s["iso_pred"] == -1]
    for _, row in anom.iterrows():
        beat = int(row["start_beat"])
        if beat < len(rr_arr):
            ax1.axvspan(beat, min(beat+20, len(rr_arr)), alpha=0.15, color="red")

ax1.set_title("RR interval series — shaded = Isolation Forest anomalies", fontsize=11)
ax1.set_xlabel("Beat number"); ax1.set_ylabel("RR interval (ms)")
ax1.legend(); ax1.set_ylim(400, 1400)

# ── (b) Anomaly score distribution ───────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
for session, color in COLORS.items():
    d = all_feat[all_feat["session"] == session]["iso_score"]
    ax2.hist(d, bins=15, color=color, alpha=0.6, label=session, edgecolor="white")
ax2.axvline(0, color="red", lw=1, ls="--", label="anomaly threshold")
ax2.set_title("Isolation Forest score\n(negative = anomalous)", fontsize=11)
ax2.set_xlabel("Score"); ax2.set_ylabel("Count"); ax2.legend()

# ── (c) Poincaré + DBSCAN ────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
unique_c = sorted(set(poi_clusters))
for c in unique_c:
    mask  = poi_clusters == c
    color = "lightgray" if c == -1 else CLUSTER_COLORS[c % len(CLUSTER_COLORS)]
    label = f"Noise" if c == -1 else f"Cluster {c}"
    ax3.scatter(all_poi[mask, 0], all_poi[mask, 1],
                c=color, s=18, alpha=0.6, label=label, edgecolors="none")
ax3.plot([400, 1400], [400, 1400], "k--", lw=0.8, alpha=0.4)
ax3.set_title("Poincaré plot — DBSCAN clusters", fontsize=11)
ax3.set_xlabel("RR[n] (ms)"); ax3.set_ylabel("RR[n+1] (ms)")
ax3.set_xlim(400, 1400); ax3.set_ylim(400, 1400)
ax3.legend(fontsize=8)

# ── (d) Poincaré coloured by session ─────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
for session, color in COLORS.items():
    mask = poi_labels_session == session
    ax4.scatter(all_poi[mask, 0], all_poi[mask, 1],
                c=color, s=18, alpha=0.5, label=session, edgecolors="none")
ax4.plot([400, 1400], [400, 1400], "k--", lw=0.8, alpha=0.4)
ax4.set_title("Poincaré plot — coloured by session", fontsize=11)
ax4.set_xlabel("RR[n] (ms)"); ax4.set_ylabel("RR[n+1] (ms)")
ax4.set_xlim(400, 1400); ax4.set_ylim(400, 1400)
ax4.legend()

# ── (e) Beat morphology cluster mean templates ────────────────────────────
ax5 = fig.add_subplot(gs[1, 2])
x_beat = np.linspace(0, 1, BEAT_SAMPLES)
for c in range(3):
    mask  = beat_clusters == c
    mean_t = all_temps[mask].mean(axis=0)
    std_t  = all_temps[mask].std(axis=0)
    color  = CLUSTER_COLORS[c]
    ax5.plot(x_beat, mean_t, color=color, lw=2, label=f"Cluster {c} (n={mask.sum()})")
    ax5.fill_between(x_beat, mean_t - std_t, mean_t + std_t, color=color, alpha=0.15)
ax5.set_title("PPG beat morphology clusters\n(mean ± std)", fontsize=11)
ax5.set_xlabel("Normalised beat time"); ax5.set_ylabel("Normalised amplitude")
ax5.legend(fontsize=8)

# ── (f) PCA embedding ────────────────────────────────────────────────────
ax6 = fig.add_subplot(gs[2, 0])
for session, color in COLORS.items():
    mask = all_feat["session"] == session
    ax6.scatter(all_feat.loc[mask, "pca1"], all_feat.loc[mask, "pca2"],
                c=color, s=40, alpha=0.7, label=session, edgecolors="white", lw=0.4)
# mark anomalies
anom_mask = all_feat["iso_pred"] == -1
ax6.scatter(all_feat.loc[anom_mask, "pca1"], all_feat.loc[anom_mask, "pca2"],
            marker="x", color="red", s=60, lw=1.5, label="Anomaly", zorder=5)
ax6.set_title(f"PCA embedding (PC1={pca2.explained_variance_ratio_[0]*100:.0f}%, "
              f"PC2={pca2.explained_variance_ratio_[1]*100:.0f}%)", fontsize=11)
ax6.set_xlabel("PC1"); ax6.set_ylabel("PC2"); ax6.legend(fontsize=8)

# ── (g) Label propagation results ────────────────────────────────────────
ax7 = fig.add_subplot(gs[2, 1])
LP_COLORS = {0: "#3B8BD4", 1: "#1D9E75", 2: "#E8593C"}
for c, name in CLASS_NAMES.items():
    mask = all_feat["label_propagated"] == c
    ax7.scatter(all_feat.loc[mask, "pca1"], all_feat.loc[mask, "pca2"],
                c=LP_COLORS[c], s=40, alpha=0.7, label=name,
                edgecolors="white", lw=0.4)
# mark anchor points
labelled_mask = labels_manual != -1
ax7.scatter(X_pca[labelled_mask, 0], X_pca[labelled_mask, 1],
            marker="*", s=180, c="gold", edgecolors="black", lw=0.7,
            label="Manual anchors", zorder=6)
ax7.set_title("Semi-supervised label propagation", fontsize=11)
ax7.set_xlabel("PC1"); ax7.set_ylabel("PC2"); ax7.legend(fontsize=8)

# ── (h) Label propagation confidence ─────────────────────────────────────
ax8 = fig.add_subplot(gs[2, 2])
conf_pre  = all_feat[all_feat["session"] == "PRE"]["label_confidence"].values
conf_post = all_feat[all_feat["session"] == "POST"]["label_confidence"].values
ax8.hist(conf_pre,  bins=15, color=COLORS["PRE"],  alpha=0.6, label="PRE",  edgecolor="white")
ax8.hist(conf_post, bins=15, color=COLORS["POST"], alpha=0.6, label="POST", edgecolor="white")
ax8.set_title("Label propagation confidence", fontsize=11)
ax8.set_xlabel("Max class probability"); ax8.set_ylabel("Count"); ax8.legend()

fig.suptitle("Unsupervised & Semi-Supervised Results\n"
             "Pre (12:18) vs Post (12:38)  |  2025-12-26",
             fontsize=13, fontweight="bold", y=1.01)

plt.savefig("ppg_analysis_results.png", dpi=150, bbox_inches="tight",
            facecolor="white")
print("  Saved → ppg_analysis_results.png")


# ═══════════════════════════════════════════════════════════════════════════
# 11. SAVE RESULTS TABLE
# ═══════════════════════════════════════════════════════════════════════════

out_cols = ["session", "start_beat", "mean_hr", "rmssd", "sdnn", "pnn50",
            "iso_pred", "lof_pred", "iso_score", "label_propagated",
            "label_confidence", "pca1", "pca2"]
all_feat[out_cols].to_csv("ppg_analysis_results.csv", index=False, float_format="%.2f")
print("  Saved → ppg_analysis_results.csv")

print("\nDone.")