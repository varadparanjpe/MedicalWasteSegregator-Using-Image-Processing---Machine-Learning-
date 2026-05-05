"""
Optional soft-voting ensemble of EfficientNet-B0 + ResNet50.

Still no LLM — this averages softmax probabilities from two CNNs, then
argmax to produce the final class. Enable via /api/predict?use_ensemble=true
when ResNet50 weights are available.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

from config import CATEGORIES, IMG_SIZE, PIXEL_MEAN, PIXEL_STD, WEIGHTS_DIR
from models.classification_model import EfficientNetClassifier

log = logging.getLogger(__name__)


def build_resnet50(num_classes: int) -> nn.Module:
    m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


class Ensemble:
    """Soft-voting: (p_effnet + p_resnet) / 2 → argmax."""

    def __init__(self, effnet: EfficientNetClassifier,
                 resnet_weights: Path = WEIGHTS_DIR / "resnet50.pth"):
        self.effnet = effnet
        self.device = effnet.device
        self.resnet: nn.Module | None = None
        if resnet_weights.exists():
            self.resnet = build_resnet50(len(CATEGORIES)).to(self.device)
            state = torch.load(resnet_weights, map_location=self.device)
            self.resnet.load_state_dict(state, strict=False)
            self.resnet.eval()
            log.info("Ensemble: ResNet50 loaded from %s", resnet_weights)
        else:
            log.info("Ensemble: ResNet50 weights not found — ensemble falls back to EfficientNet-B0.")
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(PIXEL_MEAN, PIXEL_STD),
        ])

    @torch.no_grad()
    def classify(self, crops: List[Image.Image]) -> List[Tuple[str, float]]:
        if not crops:
            return []
        if self.resnet is None:
            return self.effnet.classify(crops)

        batch = torch.stack([self.transform(c.convert("RGB")) for c in crops]).to(self.device)
        p_eff = torch.softmax(self.effnet.model(batch), dim=1)
        p_res = torch.softmax(self.resnet(batch),      dim=1)
        p = (p_eff + p_res) / 2
        conf, idx = p.max(dim=1)
        return [(CATEGORIES[i.item()], float(c.item())) for i, c in zip(idx, conf)]
