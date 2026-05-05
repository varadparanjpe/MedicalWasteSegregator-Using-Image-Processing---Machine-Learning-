"""
FastAPI entrypoint for the Biomedical Waste AI system.

Run:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Swagger UI: http://localhost:8000/docs
"""
from __future__ import annotations

import io
import logging

from fastapi import FastAPI, File, HTTPException, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from database import db
import json as _json
import sqlite3

from schemas import (
    AblationRow, Base64PredictRequest, ComparisonRow, ConfidenceHistResponse,
    ConfusionMatrixResponse, FailureCase, HistoryItem, MetricsResponse,
    ModelMetricsResponse, PerfStats, PredictResponse, TrainingEpoch, RocAucResponse,
)
from config import DB_PATH
from utils.image_ops import decode_b64_to_image
from utils.perf import TRACKER
from utils.pipeline import WastePipeline
from utils.simulator import simulate_all, clear_simulated_data
from utils.evaluate_real import main as _run_real_eval

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
log = logging.getLogger("biomed-waste-ai")

app = FastAPI(
    title="Biomedical Waste AI",
    description="YOLOv8 → EfficientNet-B0 pipeline with rule-based segregation, "
                "Grad-CAM explainability, confusion matrix, model comparison, and "
                "real-time latency tracking. No LLM anywhere.",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

pipeline: WastePipeline | None = None


@app.on_event("startup")
def _startup():
    global pipeline
    db.init_db()
    pipeline = WastePipeline()
    log.info("Pipeline initialised.")


# ─────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {"status": "ok", "pipeline": pipeline is not None}


# ─────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────
@app.post("/api/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    include_annotated: bool = True,
    use_ensemble: bool = Query(False, description="Soft-voting with ResNet50 if weights present"),
    explain: bool = Query(False, description="Attach Grad-CAM overlay per detection"),
):
    if pipeline is None:
        raise HTTPException(503, "Pipeline not initialised")
    try:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}") from e

    result = pipeline.run(
        img,
        include_annotated=include_annotated,
        use_ensemble=use_ensemble,
        explain=explain,
    )
    db.save_detections(result["detections"])
    return result


@app.post("/api/predict/base64", response_model=PredictResponse)
def predict_b64(req: Base64PredictRequest):
    if pipeline is None:
        raise HTTPException(503, "Pipeline not initialised")
    try:
        img = decode_b64_to_image(req.image_b64)
    except Exception as e:
        raise HTTPException(400, f"Invalid base64 image: {e}") from e

    result = pipeline.run(
        img,
        include_annotated=req.include_annotated,
        use_ensemble=req.use_ensemble,
        explain=req.explain,
    )
    db.save_detections(result["detections"])
    return result


# ─────────────────────────────────────────────────────────────
# Analytics (dashboard)
# ─────────────────────────────────────────────────────────────
@app.get("/api/history", response_model=list[HistoryItem])
def history(limit: int = 50):
    return db.recent_history(limit=min(max(limit, 1), 500))


@app.get("/api/metrics", response_model=MetricsResponse)
def metrics():
    return db.aggregate_metrics()


@app.get("/api/model-metrics", response_model=ModelMetricsResponse)
def model_metrics():
    return db.latest_model_metrics()


@app.get("/api/confusion-matrix", response_model=ConfusionMatrixResponse)
def confusion_matrix(model: str = "efficientnet_b0"):
    return db.latest_confusion_matrix(model=model)


@app.get("/api/model-comparison", response_model=list[ComparisonRow])
def model_comparison():
    return db.latest_comparison()


@app.get("/api/confidence-hist", response_model=ConfidenceHistResponse)
def confidence_hist():
    return db.latest_confidence_hist()


@app.get("/api/failures", response_model=list[FailureCase])
def failure_cases(limit: int = 24):
    return db.get_failure_cases(limit=min(max(limit, 1), 100))


@app.get("/api/ablation", response_model=list[AblationRow])
def ablation():
    return db.latest_ablation()


@app.get("/api/perf", response_model=PerfStats)
def perf():
    return TRACKER.stats()


@app.get("/api/training-history", response_model=list[TrainingEpoch])
def training_history():
    """Return epoch-by-epoch training curves (loss + accuracy) for the dashboard."""
    try:
        with sqlite3.connect(DB_PATH) as c:
            rows = c.execute(
                "SELECT epoch, train_loss, val_loss, train_acc, val_acc "
                "FROM training_history ORDER BY epoch ASC"
            ).fetchall()
        return [
            {"epoch": r[0], "train_loss": r[1], "val_loss": r[2],
             "train_acc": r[3], "val_acc": r[4]}
            for r in rows
        ]
    except Exception:
        return []


@app.get("/api/roc-auc", response_model=RocAucResponse)
def roc_auc():
    """Return per-class ROC AUC scores for the dashboard."""
    try:
        with sqlite3.connect(DB_PATH) as c:
            row = c.execute(
                "SELECT data FROM roc_auc ORDER BY id DESC LIMIT 1"
            ).fetchone()
        if row:
            return {"per_class": _json.loads(row[0])}
    except Exception:
        pass
    return {"per_class": {}}


@app.get("/api/model-comparison/per-class")
def model_comparison_per_class():
    """
    Return per-class metrics (precision, recall, F1, AUC) for ALL models.
    Shape: { model_name: { class_name: { precision, recall, f1, auc } } }
    Used by the dedicated Model Comparison page.
    """
    try:
        with sqlite3.connect(DB_PATH) as c:
            row = c.execute(
                "SELECT data FROM per_class_metrics ORDER BY id DESC LIMIT 1"
            ).fetchone()
        if row:
            return _json.loads(row[0])
    except Exception:
        pass
    return {}


# ─────────────────────────────────────────────────────────────
# Simulator — pre-trained "demo data" for the dashboard
# ─────────────────────────────────────────────────────────────
@app.post("/api/simulate")
def simulate(n_detections: int = Query(250, ge=50, le=2000)):
    """
    Populate every dashboard table with realistic "already trained and tested"
    data. Idempotent — safe to call repeatedly (it appends).
    """
    return simulate_all(n_detections=n_detections)


@app.post("/api/simulate/clear")
def simulate_clear():
    """Wipe all dashboard tables + reset perf tracker."""
    return clear_simulated_data()


# ─────────────────────────────────────────────────────────────
# Real evaluation — runs actual model on test dataset
# ─────────────────────────────────────────────────────────────
@app.post("/api/evaluate")
def evaluate_real():
    """
    Run real EfficientNet-B0 inference on every image in:
      /Applications/Programs/MLA_Project/processed_dataset/test
    Computes genuine accuracy, F1, confusion matrix, ROC-AUC,
    confidence histogram, and failure cases — writes all to DB.
    No hardcoded numbers. No LLM.
    """
    import threading
    try:
        t = threading.Thread(target=_run_real_eval, daemon=True)
        t.start()
        return {
            "status": "started",
            "message": "Real evaluation running in background. "
                       "Dashboard metrics will update in ~30-60 seconds. "
                       "Check backend logs for progress.",
        }
    except Exception as e:
        raise HTTPException(500, f"Evaluation failed to start: {e}") from e
