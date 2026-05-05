"""
Ablation study: no-augmentation vs with-augmentation.

Runs two short training loops and writes results into the API's SQLite so the
Dashboard can render the comparison table.

Usage:
    python ablation.py --data-root /path/to/processed_dataset --epochs 10
"""
from __future__ import annotations

import argparse, sys, time
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[2] / "backend"))
from config import CATEGORIES                                         # noqa: E402
from database import db                                               # noqa: E402
from models.classification_model import build_efficientnet            # noqa: E402
from common import device, eval_transform, loader, train_transform    # noqa: E402


def train_eval(data_root: Path, epochs: int, use_aug: bool, dev) -> tuple[float, float]:
    tr = loader(data_root, "train", tf=train_transform(use_aug), shuffle=True)
    va = loader(data_root, "val",   tf=eval_transform())

    m  = build_efficientnet(num_classes=len(CATEGORIES), pretrained=True).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    for ep in range(epochs):
        m.train()
        for x, y in tqdm(tr, desc=f"{'aug' if use_aug else 'no-aug'} ep {ep+1}/{epochs}"):
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            loss_fn(m(x), y).backward()
            opt.step()

    m.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in va:
            y_pred += m(x.to(dev)).argmax(1).cpu().tolist()
            y_true += y.tolist()
    return float(accuracy_score(y_true, y_pred)), float(f1_score(y_true, y_pred, average="macro"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--epochs", type=int, default=8)
    args = p.parse_args()

    dev = device()
    db.init_db()

    for use_aug, label in [(False, "no_augmentation"), (True, "with_augmentation")]:
        acc, f1 = train_eval(args.data_root, args.epochs, use_aug, dev)
        print(f"{label:<20}  acc={acc:.4f}  f1={f1:.4f}")
        db.save_ablation(label, acc, f1,
                         notes=f"EfficientNet-B0 · {args.epochs} epochs")


if __name__ == "__main__":
    main()
