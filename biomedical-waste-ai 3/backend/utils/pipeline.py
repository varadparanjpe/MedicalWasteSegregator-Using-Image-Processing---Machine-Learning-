"""
Full inference pipeline.

Camera / Image Input
  → YOLOv8 detect
  → Crop each object
  → Resize 224x224
  → EfficientNet-B0 classify  (or optional soft-ensemble with ResNet50)
  → Confidence check
  → Rule-based segregation (BIN_MAP)
  → Alert system
  → (optional) Grad-CAM for explainability
  → Persist + Return
"""
from __future__ import annotations

import io
import logging
import time
from typing import List

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from config import CATEGORIES, IMG_SIZE, PIXEL_MEAN, PIXEL_STD
from models.classification_model import EfficientNetClassifier
from models.detection_model import YOLODetector
from models.ensemble import Ensemble
from models.gradcam import GradCAM, overlay_heatmap
from utils.image_ops import crop_bbox, encode_image_to_b64, render_annotated
from utils.perf import TRACKER
from utils.rules import assign_bin, compute_alerts

log = logging.getLogger(__name__)


class WastePipeline:
    """Singleton orchestrator tying detector + classifier + rules together."""

    def __init__(self):
        self.detector = YOLODetector()
        self.classifier = EfficientNetClassifier()
        self.ensemble = Ensemble(self.classifier)

        # Grad-CAM attached to the last conv feature block of EfficientNet-B0
        try:
            target = self.classifier.model.features[-1]
            self.gradcam = GradCAM(self.classifier.model, target)
        except Exception as e:
            log.warning("Grad-CAM init failed: %s", e)
            self.gradcam = None

        self._cam_tf = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(PIXEL_MEAN, PIXEL_STD),
        ])

    # ────────────────────────────────────────────────────────────
    def run(self, image: Image.Image, include_annotated: bool = True,
            use_ensemble: bool = False, explain: bool = False) -> dict:
        t0 = time.perf_counter()
        W, H = image.size

        # 1. Detect
        dets = self.detector.detect(image)
        # 2. Crop
        crops = [crop_bbox(image, d["bbox"]) for d in dets]
        # 3-4. Classify (EfficientNet-B0 or soft-ensemble)
        if use_ensemble:
            classifications = self.ensemble.classify(crops)
        else:
            classifications = self.classifier.classify(crops)

        # 5-7. Assemble payload
        results: List[dict] = []
        for i, (det, crop, (category, cls_conf)) in enumerate(
            zip(dets, crops, classifications)
        ):
            bbox = det["bbox"]
            entry = {
                "id": i,
                "bbox": {"x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3]},
                "detection_confidence": det["conf"],
                "category": category,
                "classification_confidence": cls_conf,
                "bin": assign_bin(category),
                "alerts": compute_alerts(category, cls_conf),
                "cropped_image_b64": encode_image_to_b64(crop.resize((224, 224))),
            }
            if explain and self.gradcam is not None:
                try:
                    entry["gradcam_b64"] = self._gradcam_overlay(crop, category)
                except Exception as e:
                    log.warning("Grad-CAM overlay failed: %s", e)
            results.append(entry)

        annotated_b64 = None
        if include_annotated and results:
            annotated = render_annotated(image, results)
            annotated_b64 = encode_image_to_b64(annotated)

        elapsed = (time.perf_counter() - t0) * 1000.0
        TRACKER.record(elapsed)

        return {
            "success": True,
            "image_width": W,
            "image_height": H,
            "num_detections": len(results),
            "detections": results,
            "processing_ms": round(elapsed, 2),
            "annotated_image_b64": annotated_b64,
            "used_ensemble": use_ensemble and self.ensemble.resnet is not None,
        }

    # ────────────────────────────────────────────────────────────
    def _gradcam_overlay(self, crop: Image.Image, category: str) -> str:
        class_idx = CATEGORIES.index(category)
        x = self._cam_tf(crop.convert("RGB")).unsqueeze(0).to(self.classifier.device)
        x.requires_grad_(True)
        heat = self.gradcam(x, class_idx)        # (IMG_SIZE, IMG_SIZE) float
        vis  = overlay_heatmap(crop.resize((IMG_SIZE, IMG_SIZE)), heat, alpha=0.45)
        return encode_image_to_b64(vis)
