"""
DenseNet-121 classifier — medically-relevant baseline.

Architecture: DenseNet-121 (ImageNet pre-trained)
Params: ~7.98M  |  Dense connections between layers reduce vanishing gradient.
Weights path: backend/weights/densenet121.pth

Why DenseNet-121: used extensively in medical imaging research (CheXNet for
chest X-ray, various pathology classifiers). Its dense skip connections make
it robust on small-to-medium biomedical datasets and it's a well-understood
benchmark in the domain, giving capstone comparison academic credibility.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import densenet121, DenseNet121_Weights

from config import CATEGORIES, IMG_SIZE, PIXEL_MEAN, PIXEL_STD, WEIGHTS_DIR

log = logging.getLogger(__name__)

DENSENET_WEIGHTS = WEIGHTS_DIR / "densenet121.pth"


def build_densenet121(num_classes: int = len(CATEGORIES), pretrained: bool = True) -> nn.Module:
    """Build DenseNet-121 with a replaced classification head."""
    weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    model = densenet121(weights=weights)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    return model


class DenseNetClassifier:
    def __init__(
        self,
        weights_path: Path | str = DENSENET_WEIGHTS,
        device: str | None = None,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.categories = CATEGORIES
        self.model = build_densenet121(num_classes=len(self.categories), pretrained=True)

        weights_path = Path(weights_path)
        if weights_path.exists():
            state = torch.load(weights_path, map_location=self.device)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            self.model.load_state_dict(state, strict=False)
            log.info("DenseNet-121 weights loaded from %s", weights_path)
        else:
            log.warning(
                "DenseNet-121 weights not found at %s. "
                "Serving ImageNet-pretrained weights with random head. "
                "Run ml/training/train_densenet.py first.",
                weights_path,
            )

        self.model.to(self.device).eval()

        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=PIXEL_MEAN, std=PIXEL_STD),
        ])

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
