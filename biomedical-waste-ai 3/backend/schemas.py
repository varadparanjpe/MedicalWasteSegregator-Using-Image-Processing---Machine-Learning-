"""Pydantic response/request schemas."""
from typing import List, Optional, Any
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x1: float; y1: float; x2: float; y2: float


class Detection(BaseModel):
    id: int
    bbox: BoundingBox
    detection_confidence: float
    category: str
    classification_confidence: float
    bin: str
    alerts: List[str] = Field(default_factory=list)
    cropped_image_b64: Optional[str] = None
    gradcam_b64: Optional[str] = None


class PredictResponse(BaseModel):
    success: bool
    image_width: int
    image_height: int
    num_detections: int
    detections: List[Detection]
    processing_ms: float
    annotated_image_b64: Optional[str] = None
    used_ensemble: bool = False


class Base64PredictRequest(BaseModel):
    image_b64: str
    include_annotated: bool = True
    use_ensemble: bool = False
    explain: bool = False


class HistoryItem(BaseModel):
    id: int
    ts: str
    category: str
    bin: str
    confidence: float
    alerts: List[str]


class MetricsResponse(BaseModel):
    total_detections: int
    counts_by_category: dict
    counts_by_bin: dict
    avg_confidence: float
    hazard_rate: float
    low_confidence_count: int
    timeseries: List[dict]


class ModelMetricsResponse(BaseModel):
    accuracy: float; precision: float; recall: float; f1: float
    per_class: dict


class ConfusionMatrixResponse(BaseModel):
    labels: List[str]
    matrix: List[List[int]]
    model: str
    ts: Optional[str] = None


class ComparisonRow(BaseModel):
    model: str
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    params_m: Optional[float] = None
    inference_ms: Optional[float] = None
    notes: Optional[str] = None
    ts: Optional[str] = None


class PerfStats(BaseModel):
    p50: float; p90: float; p99: float
    mean: float; max: float
    throughput_fps: float
    samples: int


class ConfidenceHistResponse(BaseModel):
    bins: List[float]
    counts: List[int]
    correct: Optional[List[int]] = None


class FailureCase(BaseModel):
    id: int
    image_path: str
    true_label: str
    pred_label: str
    confidence: float
    reason: Optional[str] = None
    image_b64: Optional[str] = None


class AblationRow(BaseModel):
    setting: str
    accuracy: float
    f1: float
    notes: Optional[str] = None
    ts: Optional[str] = None


class TrainingEpoch(BaseModel):
    epoch: int
    train_loss: float
    val_loss: float
    train_acc: float
    val_acc: float


class RocAucResponse(BaseModel):
    per_class: dict  # class_name -> auc float
