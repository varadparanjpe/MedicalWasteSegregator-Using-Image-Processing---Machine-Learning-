"""
evaluate_real.py
────────────────
Runs the real EfficientNet-B0 model on every image inside:
  /Applications/Programs/MLA_Project/processed_dataset/test/<class_folder>/*.jpg|png

Computes genuine metrics (accuracy, precision, recall, F1, ROC-AUC,
confusion matrix, confidence histogram, failure cases) and writes them
into the project's SQLite database — replacing any previously simulated data.

Run ONCE after the backend has started at least once (so the DB schema exists):

    cd ~/Downloads/biomedical-waste-ai/biomedical-waste-ai/backend
    python utils/evaluate_real.py

No LLM. No hardcoding. All numbers come from your actual model + your actual dataset.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import sqlite3
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support, roc_auc_score,
)
from torchvision import transforms

# ── Make sure the backend package is importable ────────────────────────────────
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from config import CATEGORIES, DB_PATH, EFFICIENTNET_WEIGHTS, IMG_SIZE, PIXEL_MEAN, PIXEL_STD
from database import db

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s :: %(message)s")
log = logging.getLogger("evaluate_real")

# ── Config ─────────────────────────────────────────────────────────────────────
TEST_DIR = Path("/Applications/Programs/MLA_Project/processed_dataset/test")
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# ── Discover dataset folder → class name mapping ───────────────────────────────
def discover_class_map(test_dir: Path) -> dict[str, str]:
    """
    Map each subdirectory name to one of our 6 CATEGORIES.
    Handles both exact matches and common alternative names.
    """
    aliases = {
        # folder_name_lower : canonical CATEGORY
        "sharps":              "sharps_waste",
        "sharps_waste":        "sharps_waste",
        "infectious":          "infectious_waste",
        "infectious_waste":    "infectious_waste",
        "pathological":        "pathological_waste",
        "pathological_waste":  "pathological_waste",
        "plastic":             "plastic_recyclable",
        "plastic_recyclable":  "plastic_recyclable",
        "recyclable":          "plastic_recyclable",
        "pharmaceutical":      "pharmaceutical_waste",
        "pharmaceutical_waste":"pharmaceutical_waste",
        "general":             "general_waste",
        "general_waste":       "general_waste",
        "non_hazardous":       "general_waste",
    }
    mapping = {}
    for folder in sorted(test_dir.iterdir()):
        if not folder.is_dir():
            continue
        key = folder.name.lower().replace(" ", "_").replace("-", "_")
        canonical = aliases.get(key)
        if canonical is None:
            # Try partial match
            for alias, cat in aliases.items():
                if alias in key:
                    canonical = cat
                    break
        if canonical:
            mapping[folder.name] = canonical
            log.info("  Folder '%s' → '%s'", folder.name, canonical)
        else:
            log.warning("  Folder '%s' could not be mapped to a category — skipping", folder.name)
    return mapping


def collect_images(test_dir: Path, class_map: dict[str, str]) -> Tuple[List[Path], List[int]]:
    """Return parallel lists of image paths and their true class indices."""
    paths, labels = [], []
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for folder_name, category in class_map.items():
        class_idx = CATEGORIES.index(category)
        folder = test_dir / folder_name
        found = [p for p in folder.iterdir() if p.suffix.lower() in exts]
        log.info("  %s: %d images", folder_name, len(found))
        for p in found:
            paths.append(p)
            labels.append(class_idx)
    log.info("Total test images: %d", len(paths))
    return paths, labels


# ── Model loading ──────────────────────────────────────────────────────────────
def load_model():
    import torchvision.models as tv
    n_classes = len(CATEGORIES)

    if not EFFICIENTNET_WEIGHTS.exists():
        log.error("Weights not found: %s", EFFICIENTNET_WEIGHTS)
        sys.exit(1)

    log.info("Loading EfficientNet-B0 from %s on %s", EFFICIENTNET_WEIGHTS, DEVICE)
    model = tv.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, n_classes)

    state = torch.load(EFFICIENTNET_WEIGHTS, map_location=DEVICE)
    # Handle various checkpoint formats
    if isinstance(state, dict):
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        elif "state_dict" in state:
            state = state["state_dict"]
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        model.load_state_dict(state, strict=False)
        log.warning("Loaded weights with strict=False (some keys may be missing)")

    model.to(DEVICE).eval()
    return model


# ── Inference ──────────────────────────────────────────────────────────────────
TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(PIXEL_MEAN, PIXEL_STD),
])


@torch.no_grad()
def run_inference(model, paths: List[Path]) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """
    Returns:
        preds      — (N,)  predicted class indices
        probs_all  — (N, C) softmax probabilities for all classes
        latencies  — per-image latency in ms
    """
    all_probs = []
    latencies = []

    for start in range(0, len(paths), BATCH_SIZE):
        batch_paths = paths[start:start + BATCH_SIZE]
        imgs = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                imgs.append(TF(img))
            except Exception as e:
                log.warning("Could not open %s: %s — using blank", p, e)
                imgs.append(torch.zeros(3, IMG_SIZE, IMG_SIZE))

        batch = torch.stack(imgs).to(DEVICE)
        t0 = time.perf_counter()
        logits = model(batch)
        elapsed = (time.perf_counter() - t0) * 1000.0
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        latencies.extend([elapsed / len(batch_paths)] * len(batch_paths))

        if (start // BATCH_SIZE) % 5 == 0:
            log.info("  Processed %d / %d images ...", start + len(batch_paths), len(paths))

    probs_all = np.vstack(all_probs)             # (N, C)
    preds = probs_all.argmax(axis=1)             # (N,)
    return preds, probs_all, latencies


# ── Metrics computation ────────────────────────────────────────────────────────
def compute_and_save(preds: np.ndarray, labels: np.ndarray,
                     probs_all: np.ndarray, paths: List[Path],
                     latencies: List[float]) -> None:

    labels = np.array(labels)
    n = len(labels)
    log.info("Computing metrics on %d samples ...", n)

    # ── Overall accuracy ───────────────────────────────────────────────────────
    accuracy = float(accuracy_score(labels, preds))

    # ── Precision / Recall / F1 per class (macro) ──────────────────────────────
    prec_arr, rec_arr, f1_arr, support = precision_recall_fscore_support(
        labels, preds, average=None, labels=list(range(len(CATEGORIES))), zero_division=0)

    macro_prec = float(prec_arr.mean())
    macro_rec  = float(rec_arr.mean())
    macro_f1   = float(f1_arr.mean())

    per_class = {}
    for i, cat in enumerate(CATEGORIES):
        per_class[cat] = {
            "precision": round(float(prec_arr[i]), 4),
            "recall":    round(float(rec_arr[i]),  4),
            "f1":        round(float(f1_arr[i]),   4),
            "support":   int(support[i]),
        }

    log.info("Accuracy=%.4f  Precision=%.4f  Recall=%.4f  F1=%.4f",
             accuracy, macro_prec, macro_rec, macro_f1)

    # ── ROC-AUC (one-vs-rest) ──────────────────────────────────────────────────
    roc_auc_per_class = {}
    try:
        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(labels, classes=list(range(len(CATEGORIES))))
        for i, cat in enumerate(CATEGORIES):
            try:
                auc = float(roc_auc_score(y_bin[:, i], probs_all[:, i]))
                roc_auc_per_class[cat] = round(auc, 4)
            except Exception:
                roc_auc_per_class[cat] = 0.0
        log.info("Macro ROC-AUC: %.4f", np.mean(list(roc_auc_per_class.values())))
    except Exception as e:
        log.warning("ROC-AUC computation failed: %s", e)

    # ── Confusion matrix ───────────────────────────────────────────────────────
    cm = confusion_matrix(labels, preds, labels=list(range(len(CATEGORIES))))
    log.info("Confusion matrix computed.")

    # ── Confidence histogram ───────────────────────────────────────────────────
    confs = probs_all.max(axis=1)           # confidence for predicted class
    correct_mask = (preds == labels)
    bins_edges = np.linspace(0, 1, 11)     # 10 bins: 0-0.1, 0.1-0.2, ...
    counts, _ = np.histogram(confs, bins=bins_edges)
    correct_counts, _ = np.histogram(confs[correct_mask], bins=bins_edges)
    bin_labels = [f"{bins_edges[i]:.1f}-{bins_edges[i+1]:.1f}" for i in range(len(bins_edges)-1)]

    # ── Failure cases (misclassified, capped at 50) ────────────────────────────
    misclassified_idx = np.where(~correct_mask)[0]
    # Sort by descending confidence (highest-confidence mistakes are most interesting)
    misclassified_idx = sorted(misclassified_idx, key=lambda i: -confs[i])[:50]

    failure_cases = []
    for idx in misclassified_idx:
        p = paths[idx]
        true_cat  = CATEGORIES[labels[idx]]
        pred_cat  = CATEGORIES[preds[idx]]
        conf      = float(confs[idx])
        # Thumbnail
        try:
            img = Image.open(p).convert("RGB")
            img.thumbnail((96, 96))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=70)
            b64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        except Exception:
            b64 = ""
        failure_cases.append({
            "image_path": str(p),
            "true_label": true_cat,
            "pred_label": pred_cat,
            "confidence": conf,
            "reason": f"High-conf misclassification: model predicted {pred_cat} "
                      f"with {conf:.1%} confidence",
            "image_b64": b64,
        })

    # ── Mean latency ───────────────────────────────────────────────────────────
    mean_lat = float(np.mean(latencies))
    log.info("Mean inference latency: %.2f ms/image", mean_lat)

    # ── Write everything to DB ─────────────────────────────────────────────────
    log.info("Writing metrics to DB at %s ...", DB_PATH)

    # 1. Model metrics
    db.save_model_metrics({
        "accuracy":  round(accuracy,    4),
        "precision": round(macro_prec,  4),
        "recall":    round(macro_rec,   4),
        "f1":        round(macro_f1,    4),
        "per_class": per_class,
    })

    # 2. Confusion matrix
    db.save_confusion_matrix(
        labels=CATEGORIES,
        matrix=cm.tolist(),
        model="efficientnet_b0",
    )

    # 3. Confidence histogram
    db.save_confidence_hist(
        bins=bin_labels,
        counts=counts.tolist(),
        correct=correct_counts.tolist(),
    )

    # 4. Failure cases
    db.save_failure_cases(failure_cases)

    # 5. ROC-AUC → roc_auc table
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS roc_auc (
                id   INTEGER PRIMARY KEY AUTOINCREMENT,
                ts   TEXT NOT NULL,
                data TEXT NOT NULL
            )
        """)
        from datetime import datetime
        c.execute("INSERT INTO roc_auc (ts, data) VALUES (?, ?)",
                  (datetime.utcnow().isoformat(), json.dumps(roc_auc_per_class)))

    # 6. Model comparison row for EfficientNet-B0 with real latency
    db.save_comparison_row({
        "model":        "efficientnet_b0",
        "accuracy":     round(accuracy,   4),
        "precision":    round(macro_prec, 4),
        "recall":       round(macro_rec,  4),
        "f1":           round(macro_f1,   4),
        "params_m":     5.3,
        "inference_ms": round(mean_lat,   2),
        "notes":        f"Real eval on {n} test images from processed_dataset/test",
    })

    # 7. Training history (from model checkpoint if available, else skip)
    _try_write_training_history()

    log.info("=" * 60)
    log.info("DONE. Dashboard will now show real metrics.")
    log.info("  Accuracy  : %.2f%%", accuracy * 100)
    log.info("  Macro F1  : %.4f",   macro_f1)
    log.info("  Macro AUC : %.4f",   np.mean(list(roc_auc_per_class.values())) if roc_auc_per_class else 0)
    log.info("  Test images evaluated: %d", n)
    log.info("  Failures logged: %d", len(failure_cases))
    log.info("=" * 60)


def _try_write_training_history():
    """
    If the checkpoint file contains training history (loss/acc per epoch),
    write it. Otherwise writes nothing — dashboard training-curve section
    simply stays empty (no fake data).
    """
    try:
        state = torch.load(EFFICIENTNET_WEIGHTS, map_location="cpu")
        if not isinstance(state, dict):
            return
        history = state.get("history") or state.get("train_history") or state.get("log")
        if not history:
            log.info("No training history found in checkpoint — skipping training curve.")
            return

        # Normalise to list of dicts with keys: epoch, train_loss, val_loss, train_acc, val_acc
        rows = []
        for epoch_data in history:
            rows.append((
                int(epoch_data.get("epoch", len(rows) + 1)),
                float(epoch_data.get("train_loss", 0)),
                float(epoch_data.get("val_loss",   0)),
                float(epoch_data.get("train_acc",  0)),
                float(epoch_data.get("val_acc",    0)),
            ))

        with sqlite3.connect(DB_PATH) as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS training_history (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    epoch     INTEGER NOT NULL,
                    train_loss REAL, val_loss REAL,
                    train_acc  REAL, val_acc  REAL
                )
            """)
            c.execute("DELETE FROM training_history")   # replace old
            c.executemany(
                "INSERT INTO training_history (epoch, train_loss, val_loss, train_acc, val_acc) "
                "VALUES (?,?,?,?,?)",
                rows,
            )
        log.info("Training history written: %d epochs", len(rows))
    except Exception as e:
        log.info("Could not extract training history: %s", e)


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    if not TEST_DIR.exists():
        log.error("Test dataset not found at: %s", TEST_DIR)
        sys.exit(1)

    log.info("Test dataset: %s", TEST_DIR)
    log.info("Device: %s", DEVICE)

    # 1. Init DB (create tables if not yet done)
    db.init_db()

    # 2. Discover classes
    log.info("Discovering class folders ...")
    class_map = discover_class_map(TEST_DIR)
    if not class_map:
        log.error("No valid class folders found in %s", TEST_DIR)
        sys.exit(1)

    # 3. Collect images
    log.info("Collecting images ...")
    paths, labels = collect_images(TEST_DIR, class_map)
    if not paths:
        log.error("No images found in test dataset.")
        sys.exit(1)

    # 4. Load model
    model = load_model()

    # 5. Inference
    log.info("Running inference on %d images (batch=%d, device=%s) ...",
             len(paths), BATCH_SIZE, DEVICE)
    preds, probs_all, latencies = run_inference(model, paths)

    # 6. Compute and save
    compute_and_save(preds, np.array(labels), probs_all, paths, latencies)


if __name__ == "__main__":
    main()
