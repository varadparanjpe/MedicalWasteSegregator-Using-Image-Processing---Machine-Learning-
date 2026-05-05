"""
EfficientNet-B2 classifier — scaled-up comparison within the EfficientNet family.

Architecture: EfficientNet-B2 (ImageNet pre-trained)
Params: ~9.1M vs B0's ~5.3M  |  Input: 260×260 (native), resized to 224 here
Weights path: backend/weights/efficientnet_b2.pth

Why B2: natural scale-up from B0 in the same compound-scaling family.
Shows the accuracy/latency tradeoff within EfficientNets and justifies
B0 as the production choice (B2 gains ~1-1.5% acc at ~2× inference cost).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights

from config import CATEGORIES, IMG_SIZE, PIXEL_MEAN, PIXEL_STD, WEIGHTS_DIR

log = logging.getLogger(__name__)

EFFICIENTNET_B2_WEIGHTS = WEIGHTS_DIR / "efficientnet_b2.pth"


def build_efficientnet_b2(num_classes: int = len(CATEGORIES), pretrained: bool = True) -> nn.Module:
    """Build EfficientNet-B2 with a replaced classification head."""
    weights = EfficientNet_B2_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b2(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


class EfficientNetB2Classifier:
    def __init__(
        self,
        weights_path: Path | str = EFFICIENTNET_B2_WEIGHTS,
        device: str | None = None,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.categories = CATEGORIES
        self.model = build_efficientnet_b2(num_classes=len(self.categories), pretrained=True)

        weights_path = Path(weights_path)
        if weights_path.exists():
            state = torch.load(weights_path, map_location=self.device)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            self.model.load_state_dict(state, strict=False)
            log.info("EfficientNet-B2 weights loaded from %s", weights_path)
        else:
            log.warning(
                "EfficientNet-B2 weights not found at %s. "
                "Serving ImageNet-pretrained weights with random head. "
                "Run ml/training/train_efficientnet_b2.py first.",
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
