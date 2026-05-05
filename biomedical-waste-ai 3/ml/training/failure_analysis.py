"""
Find misclassified test-set images, categorise a likely reason (low
confidence, similar classes, low brightness / blur) and persist the top-N
failures with small base64 thumbnails into the API DB.  The Dashboard
"Failure cases" grid reads directly from this table.

Usage:
    python failure_analysis.py \
        --data-root /path/to/processed_dataset \
        --weights   ../../backend/weights/efficientnet_b0.pth \
        --top 24
"""
from __future__ import annotations

import argparse, base64, io, sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import datasets

sys.path.append(str(Path(__file__).resolve().parents[2] / "backend"))
from config import CATEGORIES                                         # noqa: E402
from database import db                                               # noqa: E402
from models.classification_model import build_efficientnet            # noqa: E402
from common import device, eval_transform                             # noqa: E402


SIMILAR = {
    ("sharps_waste", "pharmaceutical_waste"),
    ("infectious_waste", "pathological_waste"),
    ("plastic_recyclable", "general_waste"),
}


def likely_reason(true_label: str, pred_label: str, conf: float, img: Image.Image) -> str:
    if conf < 0.60:
        return "Very low confidence"
    if (true_label, pred_label) in SIMILAR or (pred_label, true_label) in SIMILAR:
        return f"Visually similar classes ({pred_label.replace('_', ' ')})"
    gray = np.array(img.convert("L"))
    if gray.mean() < 60:
        return "Low lighting / under-exposed"
    if gray.std() < 25:
        return "Low contrast / blur or occlusion"
    return "Unclassified — manual review recommended"


def to_thumb_b64(img: Image.Image, size: int = 160) -> str:
    img.thumbnail((size, size))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--weights",   required=True, type=Path)
    p.add_argument("--top", type=int, default=24)
    args = p.parse_args()

    dev = device()
    model = build_efficientnet(num_classes=len(CATEGORIES), pretrained=False).to(dev)
    model.load_state_dict(torch.load(args.weights, map_location=dev), strict=False)
    model.eval()

    tf = eval_transform()
    ds = datasets.ImageFolder(args.data_root / "test")
    assert ds.classes == CATEGORIES

    failures: list[dict] = []
    with torch.no_grad():
        for path, y in ds.samples:
            img = Image.open(path).convert("RGB")
            x = tf(img).unsqueeze(0).to(dev)
            probs = torch.softmax(model(x), dim=1)[0]
            conf, idx = probs.max(0)
            pred = int(idx.item())
            if pred != y:
                failures.append({
                    "image_path": str(path),
                    "true_label": CATEGORIES[y],
                    "pred_label": CATEGORIES[pred],
                    "confidence": float(conf.item()),
                    "reason": likely_reason(CATEGORIES[y], CATEGORIES[pred], float(conf.item()), img),
                    "image_b64": to_thumb_b64(img.copy()),
                })

    failures.sort(key=lambda f: f["confidence"], reverse=True)
    failures = failures[: args.top]

    db.init_db()
    db.save_failure_cases(failures)
    print(f"Saved {len(failures)} failure cases to {db.DB_PATH}")


if __name__ == "__main__":
    main()
