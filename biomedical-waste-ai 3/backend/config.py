"""
Central configuration for the Biomedical Waste AI backend.
No LLMs. No magic values hidden elsewhere.
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = BASE_DIR / "weights"
DB_PATH = BASE_DIR / "database" / "waste.db"

# ──────────────────────────────────────────────
# Model weights
# ──────────────────────────────────────────────
YOLO_WEIGHTS = WEIGHTS_DIR / "yolov8n.pt"                 # detection only
EFFICIENTNET_WEIGHTS = WEIGHTS_DIR / "efficientnet_b0.pth"  # classifier

# Image size for EfficientNet-B0
IMG_SIZE = 224

# Dataset normalization (from MLA_PreProcessing log)
PIXEL_MEAN = [0.485, 0.456, 0.406]
PIXEL_STD = [0.229, 0.224, 0.225]

# ──────────────────────────────────────────────
# 6 WHO-aligned biomedical waste categories
# ──────────────────────────────────────────────
CATEGORIES = [
    "sharps_waste",
    "infectious_waste",
    "pathological_waste",
    "plastic_recyclable",
    "pharmaceutical_waste",
    "general_waste",
]

# ──────────────────────────────────────────────
# Rule-based segregation  (NO ML here)
# ──────────────────────────────────────────────
BIN_MAP = {
    "sharps_waste":         "White Bin",
    "infectious_waste":     "Yellow Bin",
    "pathological_waste":   "Yellow Bin",
    "plastic_recyclable":   "Red Bin",
    "pharmaceutical_waste": "Blue Bin",
    "general_waste":        "Black Bin",
}

HAZARDOUS_CATEGORIES = {"sharps_waste", "infectious_waste"}
CONFIDENCE_THRESHOLD = 0.80

# ──────────────────────────────────────────────
# YOLO settings
# ──────────────────────────────────────────────
YOLO_CONF_THRESHOLD = 0.25      # detection confidence (proposals only)
YOLO_IOU_THRESHOLD = 0.45
YOLO_MAX_DETECTIONS = 25
