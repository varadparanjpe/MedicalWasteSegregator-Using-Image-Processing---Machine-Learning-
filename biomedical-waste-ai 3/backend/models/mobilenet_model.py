"""
MobileNetV3-Large classifier — lightweight comparison model.

Architecture: MobileNetV3-Large (ImageNet pre-trained)
Replace the final classifier with a 6-class head.
Weights path: backend/weights/mobilenet_v3.pth

Why MobileNetV3: designed for edge/mobile inference; similar param count to
EfficientNet-B0 (~5.4M vs ~5.3M) but optimised for lower latency on CPU,
which is relevant for hospital-bedside or bin-mounted cameras.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

from config import CATEGORIES, IMG_SIZE, PIXEL_MEAN, PIXEL_STD, WEIGHTS_DIR

log = logging.getLogger(__name__)

MOBILENET_WEIGHTS = WEIGHTS_DIR / "mobilenet_v3.pth"


def build_mobilenet_v3(num_classes: int = len(CATEGORIES), pretrained: bool = True) -> nn.Module:
    """Build MobileNetV3-Large with a replaced classification head."""
    weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
    model = mobilenet_v3_large(weights=weights)
    # Replace the final linear layer in the classifier block
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


class MobileNetClassifier:
    def __init__(
        self,
        weights_path: Path | str = MOBILENET_WEIGHTS,
        device: str | None = None,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.categories = CATEGORIES
        self.model = build_mobilenet_v3(num_classes=len(self.categories), pretrained=True)

        weights_path = Path(weights_path)
        if weights_path.exists():
            state = torch.load(weights_path, map_location=self.device)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            self.model.load_state_dict(state, strict=False)
            log.info("MobileNetV3-Large weights loaded from %s", weights_path)
        else:
            log.warning(
                "MobileNetV3-Large weights not found at %s. "
                "Serving ImageNet-pretrained weights with random head. "
                "Run ml/training/train_mobilenet.py first.",
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
