"""
EfficientNet-B0 classifier.

This model is the ONLY source of final waste-category labels.
Input: 224x224 RGB crop produced by the YOLO → crop → resize stage.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from config import (
    EFFICIENTNET_WEIGHTS,
    IMG_SIZE,
    PIXEL_MEAN,
    PIXEL_STD,
    CATEGORIES,
)

log = logging.getLogger(__name__)


def build_efficientnet(num_classes: int = len(CATEGORIES), pretrained: bool = True) -> nn.Module:
    """Build an EfficientNet-B0 with a replaced classification head."""
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


class EfficientNetClassifier:
    def __init__(
        self,
        weights_path: Path | str = EFFICIENTNET_WEIGHTS,
        device: str | None = None,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.categories = CATEGORIES
        self.model = build_efficientnet(num_classes=len(self.categories), pretrained=True)

        weights_path = Path(weights_path)
        if weights_path.exists():
            state = torch.load(weights_path, map_location=self.device)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            self.model.load_state_dict(state, strict=False)
            log.info("EfficientNet-B0 weights loaded from %s", weights_path)
        else:
            log.warning(
                "EfficientNet weights not found at %s. "
                "Serving ImageNet-pretrained weights with randomly initialised head. "
                "Run ml/training/train_efficientnet.py first for production use.",
                weights_path,
            )

        self.model.to(self.device).eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=PIXEL_MEAN, std=PIXEL_STD),
            ]
        )

    @torch.no_grad()
    def classify(self, crops: List[Image.Image]) -> List[Tuple[str, float]]:
        if not crops:
            return []
        batch = torch.stack([self.transform(c.convert("RGB")) for c in crops]).to(self.device)
        logits = self.model(batch)
        probs = torch.softmax(logits, dim=1)
        confs, idxs = probs.max(dim=1)
        return [
            (self.categories[i.item()], float(c.item()))
            for i, c in zip(idxs, confs)
        ]
