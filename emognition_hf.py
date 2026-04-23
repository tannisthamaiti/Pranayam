"""
Emognition Dataset — Hugging Face Inference API Pipeline
=========================================================
Three approaches, all via HF API (no local GPU needed):

  A. Zero-shot  : Idefics2 / LLaVA vision-language model
  B. Embedding  : CLIP via HF feature-extraction endpoint + sklearn
  C. Fusion     : CLIP visual + physiological features combined

Requirements:
    pip install huggingface_hub pillow opencv-python
    pip install scikit-learn pandas numpy tqdm requests

Get your token: https://huggingface.co/settings/tokens
Set it:  export HF_TOKEN=hf_...
"""

import os, json, base64, time
from pathlib import Path
from io import BytesIO

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import requests

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

HF_TOKEN      = os.environ.get("HF_TOKEN", "hf_YOUR_TOKEN_HERE")
DATASET_ROOT  = Path("emognition")       # folder with P01/, P02/ ...
VIDEO_ROOT    = Path("videos")           # P01_amusement.mp4 etc.
OUTPUT_DIR    = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

EMOTIONS = [
    "amusement", "anger", "awe", "disgust",
    "enthusiasm", "fear", "liking", "sadness", "surprise"
]

FRAMES_PER_CLIP = 6   # frames sampled per video clip

# ── Hugging Face model endpoints ──────────────────────────────────────────

# Vision-Language model for zero-shot labelling
# Options (free tier, swap as needed):
#   "HuggingFaceM4/idefics2-8b"           ← strong, supports multi-image
#   "llava-hf/llava-1.5-7b-hf"            ← widely used
#   "Salesforce/blip2-opt-2.7b"            ← lighter weight
VLM_MODEL = "HuggingFaceM4/idefics2-8b"

# Vision encoder for embeddings (CLIP)
# Options:
#   "openai/clip-vit-large-patch14"        ← best quality
#   "openai/clip-vit-base-patch32"         ← faster, lighter
CLIP_MODEL = "openai/clip-vit-large-patch14"

HF_API_BASE = "https://api-inference.huggingface.co/models"

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type":  "application/json",
}


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def hf_post(model: str, payload: dict, retries: int = 4) -> dict | list:
    """POST to HF Inference API with retry + loading wait."""
    url = f"{HF_API_BASE}/{model}"
    for attempt in range(retries):
        resp = requests.post(url, headers=HEADERS, json=payload, timeout=120)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 503:
            wait = int(resp.json().get("estimated_time", 20))
            print(f"  Model loading, waiting {wait}s …")
            time.sleep(wait + 2)
        elif resp.status_code == 429:
            print(f"  Rate limited, sleeping 60s …")
            time.sleep(60)
        else:
            print(f"  HF API error {resp.status_code}: {resp.text[:200]}")
            time.sleep(5)
    raise RuntimeError(f"HF API failed after {retries} retries for {model}")


def pil_to_base64(img: Image.Image, fmt: str = "JPEG") -> str:
    """Encode a PIL image to base64 string."""
    buf = BytesIO()
    img.save(buf, format=fmt, quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def extract_frames(video_path: Path, n: int = FRAMES_PER_CLIP) -> list[Image.Image]:
    """Uniformly sample n frames from a video, return as PIL RGB images."""
    cap   = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return []
    indices = np.linspace(0, total - 1, n, dtype=int)
    frames  = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


# ─────────────────────────────────────────────
# STEP 1 — LOAD SELF-REPORT LABELS
# ─────────────────────────────────────────────

def load_labels(dataset_root: Path) -> pd.DataFrame:
    """
    Parse questionnaires.json for every participant.
    Returns DataFrame: participant | emotion_condition | target_emotion |
                       target_rating | valence | arousal | motivation
    """
    records = []
    for p_dir in sorted(dataset_root.iterdir()):
        if not p_dir.is_dir():
            continue
        qfile = p_dir / "questionnaires.json"
        if not qfile.exists():
            continue
        with open(qfile) as f:
            q = json.load(f)

        participant = p_dir.name
        for clip_name, ratings in q.get("discrete", {}).items():
            condition = clip_name.replace("_stimulus", "")
            dim       = q.get("dimensional", {}).get(clip_name, {})
            record    = {
                "participant":       participant,
                "emotion_condition": condition,
                "valence":           dim.get("valence"),
                "arousal":           dim.get("arousal"),
                "motivation":        dim.get("motivation"),
                "target_emotion":    max(ratings, key=ratings.get) if ratings else condition,
                "target_rating":     max(ratings.values()) if ratings else None,
            }
            records.append(record)

    df = pd.DataFrame(records)
    print(f"Labels loaded: {len(df)} rows, {df['participant'].nunique()} participants")
    return df


# ─────────────────────────────────────────────
# APPROACH A — ZERO-SHOT VLM (Idefics2 / LLaVA)
# ─────────────────────────────────────────────

def build_vlm_prompt(emotion_list: list[str]) -> str:
    emotions_str = ", ".join(emotion_list)
    return (
        "You are analysing frames from an emotion elicitation study. "
        "The participant watched a short film clip and their upper body "
        "and face were recorded.\n\n"
        f"Choose exactly ONE emotion from this list: {emotions_str}\n\n"
        "Respond ONLY as valid JSON, no other text:\n"
        '{"predicted": "<emotion>", "confidence": "high|medium|low", '
        '"reasoning": "<one sentence max>"}'
    )


def zero_shot_single_clip(frames: list[Image.Image]) -> dict:
    """
    Sends frames to Idefics2 via HF Inference API.
    Idefics2 accepts interleaved image+text in the 'inputs' field.
    """
    # Encode frames as base64 data URIs
    image_blocks = []
    for img in frames:
        b64 = pil_to_base64(img.resize((336, 336)))
        image_blocks.append(f"data:image/jpeg;base64,{b64}")

    # Idefics2 chat format
    payload = {
        "inputs": {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        [{"type": "image", "url": url} for url in image_blocks]
                        + [{"type": "text", "text": build_vlm_prompt(EMOTIONS)}]
                    )
                }
            ]
        },
        "parameters": {
            "max_new_tokens": 120,
            "temperature":    0.1,
            "return_full_text": False,
        }
    }

    try:
        result = hf_post(VLM_MODEL, payload)
        # result is usually [{"generated_text": "..."}]
        raw = result[0]["generated_text"] if isinstance(result, list) else str(result)
        # extract JSON substring
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        return json.loads(raw[start:end]) if start != -1 else {
            "predicted": "unknown", "confidence": "low", "reasoning": raw
        }
    except Exception as e:
        return {"predicted": "error", "confidence": "low", "reasoning": str(e)}


def run_zero_shot_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Zero-shot VLM"):
        vid = VIDEO_ROOT / f"{row['participant']}_{row['emotion_condition']}.mp4"
        if not vid.exists():
            continue

        frames = extract_frames(vid, n=4)   # use 4 frames to stay within payload limits
        if not frames:
            continue

        pred = zero_shot_single_clip(frames)
        results.append({
            "participant":       row["participant"],
            "emotion_condition": row["emotion_condition"],
            "ground_truth":      row["target_emotion"],
            "predicted":         pred.get("predicted", "unknown"),
            "confidence":        pred.get("confidence", ""),
            "reasoning":         pred.get("reasoning", ""),
        })
        time.sleep(1.0)   # be courteous to the free tier

    out = pd.DataFrame(results)
    out.to_csv(OUTPUT_DIR / "zero_shot_results.csv", index=False)

    valid = out[out["predicted"].isin(EMOTIONS)]
    acc   = (valid["ground_truth"] == valid["predicted"]).mean()
    print(f"\nZero-shot accuracy (valid predictions): {acc:.1%}")
    print(classification_report(valid["ground_truth"], valid["predicted"],
                                labels=EMOTIONS, zero_division=0))
    return out


# ─────────────────────────────────────────────
# APPROACH B — CLIP EMBEDDINGS VIA HF API
# ─────────────────────────────────────────────

def get_clip_embedding(image: Image.Image) -> np.ndarray | None:
    """
    Calls HF feature-extraction endpoint for CLIP.
    Returns a 1-D numpy array (768-d for ViT-Large).
    """
    b64 = pil_to_base64(image.resize((224, 224)))
    payload = {"inputs": {"image": b64}}   # base64 image input

    try:
        result = hf_post(CLIP_MODEL, payload)
        # result shape from HF: list of floats  OR  [[floats]]
        if isinstance(result, list):
            arr = np.array(result)
            return arr.flatten()
        return None
    except Exception as e:
        print(f"  Embedding error: {e}")
        return None


def extract_all_clip_embeddings(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts per-clip CLIP embeddings (mean of frame embeddings).
    Returns X (N, D), y (N,), groups (N,) for leave-one-participant-out CV.
    """
    X_list, y_list, groups = [], [], []
    le = LabelEncoder().fit(EMOTIONS)

    cache_X = OUTPUT_DIR / "hf_clip_X.npy"
    cache_y = OUTPUT_DIR / "hf_clip_y.npy"
    cache_g = OUTPUT_DIR / "hf_clip_groups.npy"

    if cache_X.exists():
        print("Loading cached embeddings …")
        return np.load(cache_X), np.load(cache_y), np.load(cache_g, allow_pickle=True)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="CLIP embeddings"):
        vid = VIDEO_ROOT / f"{row['participant']}_{row['emotion_condition']}.mp4"
        if not vid.exists():
            continue

        frames = extract_frames(vid)
        if not frames:
            continue

        frame_embeddings = []
        for frame in frames:
            emb = get_clip_embedding(frame)
            if emb is not None:
                frame_embeddings.append(emb)
            time.sleep(0.3)   # rate-limit buffer

        if not frame_embeddings:
            continue

        clip_emb = np.mean(frame_embeddings, axis=0)   # average across frames
        label    = le.transform([row["target_emotion"]])[0]

        X_list.append(clip_emb)
        y_list.append(label)
        groups.append(row["participant"])

    X      = np.vstack(X_list)
    y      = np.array(y_list)
    groups = np.array(groups)

    np.save(cache_X, X)
    np.save(cache_y, y)
    np.save(cache_g, groups)
    print(f"Embeddings saved: {X.shape}")
    return X, y, groups


def run_clip_classifier(df: pd.DataFrame) -> None:
    X, y, groups = extract_all_clip_embeddings(df)
    le           = LabelEncoder().fit(EMOTIONS)

    # ── Leave-one-participant-out cross-validation ──
    # Better than random k-fold for this kind of study
    logo     = LeaveOneGroupOut()
    all_true, all_pred = [], []

    for train_idx, val_idx in tqdm(logo.split(X, y, groups),
                                    total=len(np.unique(groups)),
                                    desc="LOGO-CV"):
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("lr",     LogisticRegression(max_iter=1000, C=1.0, random_state=42))
        ])
        clf.fit(X[train_idx], y[train_idx])
        preds = clf.predict(X[val_idx])
        all_true.extend(y[val_idx])
        all_pred.extend(preds)

    acc = (np.array(all_true) == np.array(all_pred)).mean()
    print(f"\nCLIP LOGO-CV accuracy: {acc:.1%}")
    print(classification_report(
        all_true, all_pred,
        target_names=le.inverse_transform(range(len(EMOTIONS))),
        zero_division=0
    ))

    cm    = confusion_matrix(all_true, all_pred)
    names = le.inverse_transform(range(len(EMOTIONS)))
    pd.DataFrame(cm, index=names, columns=names).to_csv(
        OUTPUT_DIR / "clip_confusion_matrix.csv"
    )
    print("Confusion matrix → outputs/clip_confusion_matrix.csv")


# ─────────────────────────────────────────────
# APPROACH C — MULTIMODAL FUSION
# ─────────────────────────────────────────────

def extract_physio_features(signal_path: Path) -> np.ndarray:
    """
    Reads one Emognition JSON signal file and returns a 9-d feature vector:
    mean, std, min, max, range, RMS, mean|diff|, skewness, kurtosis-proxy
    """
    if not signal_path.exists():
        return np.zeros(9)
    with open(signal_path) as f:
        data = json.load(f)
    s = np.array(data.get("signal", data.get("data", [])), dtype=float)
    if len(s) < 2:
        return np.zeros(9)
    d = np.diff(s)
    return np.array([
        s.mean(), s.std(), s.min(), s.max(),
        s.max() - s.min(),
        np.sqrt((s**2).mean()),         # RMS
        np.abs(d).mean(),               # mean absolute first derivative
        (((s - s.mean()) / (s.std() + 1e-8))**3).mean(),   # skewness
        (((s - s.mean()) / (s.std() + 1e-8))**4).mean(),   # kurtosis proxy
    ])


PHYSIO_SIGNALS = [
    ("empatica", "bvp"),
    ("empatica", "eda"),
    ("empatica", "skt"),
    ("samsung",  "hr"),
]


def build_physio_matrix(df: pd.DataFrame) -> np.ndarray:
    rows = []
    for _, row in df.iterrows():
        p_dir    = DATASET_ROOT / row["participant"]
        clip     = row["emotion_condition"]
        features = []
        for device, sig in PHYSIO_SIGNALS:
            # e.g. amusement_stimulus_empatica_bvp.json
            path = p_dir / f"{clip}_stimulus_{device}_{sig}.json"
            features.append(extract_physio_features(path))
        rows.append(np.concatenate(features))
    return np.vstack(rows)


def run_fusion_pipeline(df: pd.DataFrame) -> None:
    cache_X = OUTPUT_DIR / "hf_clip_X.npy"
    cache_y = OUTPUT_DIR / "hf_clip_y.npy"
    cache_g = OUTPUT_DIR / "hf_clip_groups.npy"

    if not cache_X.exists():
        print("Run run_clip_classifier() first to cache embeddings.")
        return

    X_visual = np.load(cache_X)
    y        = np.load(cache_y)
    groups   = np.load(cache_g, allow_pickle=True)
    le       = LabelEncoder().fit(EMOTIONS)

    # Rebuild physio matrix aligned to cached rows
    # (only rows where video existed — mirror that filter)
    cached_rows = []
    for _, row in df.iterrows():
        vid = VIDEO_ROOT / f"{row['participant']}_{row['emotion_condition']}.mp4"
        if vid.exists():
            cached_rows.append(row)
    df_cached = pd.DataFrame(cached_rows).reset_index(drop=True)

    print("Extracting physiological features …")
    X_physio = build_physio_matrix(df_cached)

    X_fused = np.hstack([X_visual, X_physio])
    print(f"Fused shape: {X_fused.shape}  "
          f"(CLIP {X_visual.shape[1]}d + physio {X_physio.shape[1]}d)")

    logo     = LeaveOneGroupOut()
    all_true, all_pred = [], []

    for train_idx, val_idx in tqdm(logo.split(X_fused, y, groups),
                                    total=len(np.unique(groups)),
                                    desc="Fusion LOGO-CV"):
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("lr",     LogisticRegression(max_iter=1000, C=1.0, random_state=42))
        ])
        clf.fit(X_fused[train_idx], y[train_idx])
        preds = clf.predict(X_fused[val_idx])
        all_true.extend(y[val_idx])
        all_pred.extend(preds)

    acc = (np.array(all_true) == np.array(all_pred)).mean()
    print(f"\nFusion LOGO-CV accuracy: {acc:.1%}")
    print(classification_report(
        all_true, all_pred,
        target_names=le.inverse_transform(range(len(EMOTIONS))),
        zero_division=0
    ))


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    df = load_labels(DATASET_ROOT)

    # ── Pick which pipeline to run ────────────────────────────────────────

    # A) Zero-shot — send frames to Idefics2 / LLaVA, no training needed
    # run_zero_shot_pipeline(df)

    # B) CLIP embeddings → logistic regression, leave-one-participant-out CV
    # run_clip_classifier(df)

    # C) CLIP visual + physiological signals fused, same CV scheme
    # run_fusion_pipeline(df)

    print("Uncomment whichever pipeline you want to run.")
    print("Outputs →", OUTPUT_DIR.resolve())
