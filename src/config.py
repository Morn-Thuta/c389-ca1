"""Shared constants: class names, training config, thresholds and Drive paths (PRD §4, §6, §7)."""

import os
from pathlib import Path

# --- Classes (positional: never reorder, the lecturer notebooks check 0 and 1) ---
CLASS_NAMES = ["person_left", "person_right", "bicycle"]
BASELINE_CLASSES = CLASS_NAMES[:2]
TWIST_CLASSES = CLASS_NAMES[:3]
PERSON_CLASSES = {"person_left", "person_right"}

# --- Paths ---
# Colab default; set PROJECT_ROOT to run locally.
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", "/content/drive/MyDrive/C389"))
RUNS_DIR = PROJECT_ROOT / "runs"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DATASETS_DIR = PROJECT_ROOT / "datasets"
SPLITS = ["train", "valid", "test"]

# --- Training (PRD §6) ---
BASE_WEIGHTS = "yolo26n.pt"
TRAIN_CONFIG = {
    "imgsz": 640,
    "epochs": 100,
    "batch": 16,
    "fliplr": 0.0,  # flipping reverses direction but keeps the old label
    "seed": 0,
}
RUN_BASELINE = "baseline_2cls"
RUN_TWIST = "twist_3cls"

# --- Inference ---
CONF_THRESHOLD = 0.40  # default; must be justified by the validation threshold sweep

# --- Person-bike linking (PRD §7.2); tune on validation only ---
LINK_LOWER_FRACTION = 0.60  # bottom 60% of the person box
LINK_MIN_OVERLAP = 0.25     # area(lower ∩ bike) / area(bike)
LINK_X_EXPAND = 0.10        # expand bike x-range by 10% of its width

# --- Data checks (PRD §5.4) ---
MIN_BOX_HEIGHT_FRAC = 0.10  # flag boxes shorter than 10% of image height
DEDUPE_MAX_HAMMING = 6      # phash distance counted as near-duplicate
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
