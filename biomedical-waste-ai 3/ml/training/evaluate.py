"""
Standalone evaluation: EfficientNet-B0 on test split.
Prints metrics and writes them into the FastAPI DB so the Dashboard updates.
"""
import argparse, sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "backend"))
from config import CATEGORIES, IMG_SIZE, PIXEL_MEAN, PIXEL_STD        # noqa: E402
from models.classification_model import build_efficientnet            # noqa: E402
from train_efficientnet import evaluate, persist_model_metrics         # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--weights", required=True, type=Path)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(PIXEL_MEAN, PIXEL_STD),
    ])
    loader = DataLoader(datasets.ImageFolder(args.data_root / "test", transform=tf),
                        batch_size=32, shuffle=False, num_workers=4)

    device = torch.device(args.device)
    model = build_efficientnet(num_classes=len(CATEGORIES), pretrained=False).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))

    acc, pr, rc, f1, per_class = evaluate(model, loader, device)
    metrics = {"accuracy": acc, "precision": pr, "recall": rc, "f1": f1, "per_class": per_class}
    print(metrics)
    persist_model_metrics(metrics)


if __name__ == "__main__":
    main()
