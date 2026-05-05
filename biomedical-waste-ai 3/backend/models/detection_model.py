"""
YOLOv8 detection wrapper.

IMPORTANT: This model is used ONLY for object detection (bounding boxes +
objectness). The class label produced by YOLO is discarded — final
classification is performed by EfficientNet-B0 on each crop.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

from config import (
    YOLO_WEIGHTS,
    YOLO_CONF_THRESHOLD,
    YOLO_IOU_THRESHOLD,
    YOLO_MAX_DETECTIONS,
)

log = logging.getLogger(__name__)


class YOLODetector:
    """Thin wrapper around Ultralytics YOLOv8."""

    def __init__(self, weights_path: Path | str = YOLO_WEIGHTS):
        from ultralytics import YOLO

        weights_path = Path(weights_path)
        if not weights_path.exists():
            log.warning(
                "YOLO weights not found at %s — falling back to pretrained 'yolov8n.pt'.",
                weights_path,
            )
            weights_path = "yolov8n.pt"

        self.model = YOLO(str(weights_path))
        log.info("YOLOv8 detector loaded from %s", weights_path)

    def detect(self, image: Image.Image) -> List[dict]:
        """
        Run detection on a PIL image and return a list of boxes.

        Returns:
            [{'bbox': (x1,y1,x2,y2), 'conf': float}, ...]
        """
        arr = np.array(image.convert("RGB"))
        results = self.model.predict(
            source=arr,
            conf=YOLO_CONF_THRESHOLD,
            iou=YOLO_IOU_THRESHOLD,
            max_det=YOLO_MAX_DETECTIONS,
            verbose=False,
        )

        detections: List[dict] = []
        if not results:
            return detections

        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            # Fallback: treat the entire image as one region so the pipeline
            # still produces a classification. This is safe because the final
            # label always comes from EfficientNet-B0.
            w, h = image.size
            return [{"bbox": (0.0, 0.0, float(w), float(h)), "conf": 1.0}]

        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), c in zip(xyxy, confs):
            detections.append(
                {
                    "bbox": (float(x1), float(y1), float(x2), float(y2)),
                    "conf": float(c),
                }
            )
        return detections
