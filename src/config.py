"""Shared constants: class names, training config, thresholds and Drive paths (PRD §4, §6, §7)."""

import os
from pathlib import Path

# --- Classes (positional: never reorder, the lecturer notebooks check 0 and 1) ---
# Twist 2 (micromobility crossers): a rider and their bike or scooter share one
# wheeled_* box, with the direction they are travelling.
CLASS_NAMES = ["person_left", "person_right", "wheeled_left", "wheeled_right"]
BASELINE_CLASSES = CLASS_NAMES[:2]
TWIST_CLASSES = CLASS_NAMES
PERSON_CLASSES = {"person_left", "person_right"}
WHEELED_CLASSES = {"wheeled_left", "wheeled_right"}
# Older exports labelled the bike on its own; src/data/to_wheeled.py converts them.
LEGACY_BICYCLE = "bicycle"

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
RUN_TWIST = "twist_4cls"

# --- Inference ---
CONF_THRESHOLD = 0.40  # default; must be justified by the validation threshold sweep
# Square size the photo is stretched to before the model sees it, matching
# Roboflow's "Stretch to 512x512" export the current weights were trained on.
# Set to None for weights trained on an export that keeps the photo's shape
# ("Fit within"): the model then letterboxes as normal.
INPUT_STRETCH = 512
# Size the model runs at: the training imgsz (the 512 squares were enlarged to
# 640 in training, and validation mAP50 is 0.876 at 640 against 0.859 at 512).
INFER_IMGSZ = TRAIN_CONFIG["imgsz"]
# Two boxes of the same kind overlapping this much are one crosser (YOLO26 is
# NMS-free and can repeat a box); the more confident one is kept.
DUPLICATE_IOU = 0.70

# --- Person-bike linking (src/linking.py): used to turn a rider + bike into one
# wheeled box when converting old labels and when pre-labelling ---
LINK_LOWER_FRACTION = 0.60  # bottom 60% of the person box
LINK_MIN_OVERLAP = 0.25     # area(lower ∩ bike) / area(bike)
LINK_X_EXPAND = 0.10        # expand bike x-range by 10% of its width
# Walking or carrying a bike also counts as wheeled (team rule): a bike beside the
# person's lower body, within this share of the person's width, is theirs.
LINK_BESIDE_GAP = 0.25
LINK_BESIDE_MIN_SPAN = 0.50  # share of the bike's height level with the lower body

# --- Data checks (PRD §5.4) ---
MIN_BOX_HEIGHT_FRAC = 0.10  # flag boxes shorter than 10% of image height
DEDUPE_MAX_HAMMING = 6      # phash distance counted as near-duplicate
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
