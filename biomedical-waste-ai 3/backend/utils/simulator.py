"""
simulator.py — Real data seeder.

Runs the actual EfficientNet-B0 + YOLOv8 pipeline on test images from:
  /Applications/Programs/MLA_Project/processed_dataset/test

Populates:
  - detections table  (real pipeline output per image)
  - model_metrics     (real accuracy / F1 / precision / recall)
  - confusion_matrix  (real 6x6 from test set)
  - roc_auc           (real per-class AUC)
  - confidence_hist   (real distribution)
  - failure_cases     (real misclassified images with thumbnails)

No hardcoded metrics. No fake numbers. No LLM.
"""
from __future__ import annotations

import sqlite3
import logging
from pathlib import Path

from config import DB_PATH
from database import db
from utils.perf import TRACKER

log = logging.getLogger(__name__)

TEST_DIR = Path("/Applications/Programs/MLA_Project/processed_dataset/test")


def simulate_all(n_detections: int = 250) -> dict:
    """
    Entry point called by POST /api/simulate.

    Runs evaluate_real.main() which:
      1. Loads the real EfficientNet-B0 weights
      2. Runs real forward passes on every image in TEST_DIR
      3. Writes real metrics to the DB

    Then seeds the detections table by running the full YOLOv8+EfficientNet
    pipeline on a sample of test images so the dashboard history table
    and timeseries chart have real data.
    """
    db.init_db()

    if not TEST_DIR.exists():
        return {
            "seeded": False,
            "error": f"Test dataset not found at {TEST_DIR}. "
                     "Please ensure the processed_dataset is at that path.",
        }

    log.info("Running real evaluation on test dataset at %s", TEST_DIR)

    try:
        from utils.evaluate_real import main as run_eval
        run_eval()
    except Exception as e:
        log.error("evaluate_real failed: %s", e)
        return {"seeded": False, "error": str(e)}

    try:
        _seed_real_detections()
    except Exception as e:
        log.warning("Detection history seeding failed (non-fatal): %s", e)

    with sqlite3.connect(DB_PATH) as c:
        n_det = c.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
        n_mm  = c.execute("SELECT COUNT(*) FROM model_metrics").fetchone()[0]
        n_cm  = c.execute("SELECT COUNT(*) FROM confusion_matrix").fetchone()[0]

    return {
        "seeded": True,
        "detections": n_det,
        "model_metrics_rows": n_mm,
        "confusion_matrix_rows": n_cm,
        "source": str(TEST_DIR),
    }


def _seed_real_detections(max_images: int = 300) -> None:
    """
    Run the real YOLOv8 + EfficientNet-B0 pipeline on up to max_images
    test images and save detections with timestamps spread across last 7 days.
    """
    import json
    import random
    from datetime import datetime, timedelta
    from PIL import Image

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    all_images = []
    for folder in TEST_DIR.iterdir():
        if folder.is_dir():
            all_images.extend(p for p in folder.iterdir() if p.suffix.lower() in exts)

    if not all_images:
        log.warning("No images found in %s for detection seeding", TEST_DIR)
        return

    rng = random.Random(42)
    sample = rng.sample(all_images, min(max_images, len(all_images)))

    from utils.pipeline import WastePipeline
    pipeline = WastePipeline()

    now = datetime.utcnow()
    rows = []

    for img_path in sample:
        try:
            img = Image.open(img_path).convert("RGB")
            result = pipeline.run(img, include_annotated=False)
            detections = result.get("detections", [])
            if not detections:
                continue
            offset_secs = rng.randint(0, 7 * 24 * 3600)
            ts = (now - timedelta(seconds=offset_secs)).isoformat()
            for d in detections:
                rows.append((
                    ts,
                    d["category"],
                    d["bin"],
                    float(d["classification_confidence"]),
                    float(d["detection_confidence"]),
                    json.dumps(d.get("alerts", [])),
                    json.dumps(d["bbox"]),
                ))
        except Exception as e:
            log.debug("Skipping %s: %s", img_path.name, e)
            continue

    if rows:
        with sqlite3.connect(DB_PATH) as c:
            c.executemany(
                "INSERT INTO detections "
                "(ts, category, bin, confidence, det_conf, alerts, bbox) "
                "VALUES (?,?,?,?,?,?,?)",
                rows,
            )
        log.info("Seeded %d real detections from %d images", len(rows), len(sample))
    else:
        log.warning("No detections produced from sample images")


def clear_simulated_data() -> dict:
    """Wipe all dashboard tables and reset the perf tracker."""
    tables = (
        "detections", "model_metrics", "confusion_matrix",
        "model_comparison", "confidence_hist", "failure_cases",
        "ablation_results", "training_history", "roc_auc",
        "per_class_metrics",
    )
    with sqlite3.connect(DB_PATH) as c:
        for table in tables:
            try:
                c.execute(f"DELETE FROM {table}")
            except Exception:
                pass

    TRACKER._buf.clear()
    return {"cleared": True}
