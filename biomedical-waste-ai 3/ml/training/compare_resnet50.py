"""
Train ResNet50 as a comparison baseline to EfficientNet-B0.
Writes accuracy/F1/params/inference_ms into model_comparison so the
Dashboard renders a side-by-side table.

Usage:
    python compare_resnet50.py --data-root /path/to/processed_dataset \
        --out ../../backend/weights/resnet50.pth --epochs 20
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch, torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

sys.path.append(str(Path(__file__).resolve().parents[2] / "backend"))
from config import CATEGORIES                              # noqa: E402
from database import db                                    # noqa: E402
from common import device, eval_transform, loader, train_transform  # noqa: E402


def build_resnet50(num_classes: int):
    m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


@torch.no_grad()
def evaluate(model, dl, dev):
    model.eval()
    y_true, y_pred = [], []
    times = []
    for x, y in dl:
        x = x.to(dev)
        t0 = time.perf_counter()
        out = model(x)
        times.append((time.perf_counter() - t0) * 1000 / x.size(0))
        y_pred += out.argmax(1).cpu().tolist(); y_true += y.tolist()
    acc = accuracy_score(y_true, y_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return acc, pr, rc, f1, y_true, y_pred, float(np.mean(times))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    args = p.parse_args()

    dev = device()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    tr = loader(args.data_root, "train", args.batch_size, tf=train_transform(True), shuffle=True)
    va = loader(args.data_root, "val",   args.batch_size, tf=eval_transform())
    te = loader(args.data_root, "test",  args.batch_size, tf=eval_transform())

    m = build_resnet50(len(CATEGORIES)).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    best_f1 = -1.0

    for ep in range(args.epochs):
        m.train()
        for x, y in tqdm(tr, desc=f"resnet epoch {ep+1}"):
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(); loss_fn(m(x), y).backward(); opt.step()
        acc, *_, f1, *_, = evaluate(m, va, dev)
        print(f"[epoch {ep+1}] val acc={acc:.4f} f1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1; torch.save(m.state_dict(), args.out)

    # Test + persist
    m.load_state_dict(torch.load(args.out, map_location=dev))
    acc, pr, rc, f1, y_true, y_pred, mean_ms = evaluate(m, te, dev)

    db.init_db()
    params_m = sum(p_.numel() for p_ in m.parameters()) / 1e6
    db.save_comparison_row({
        "model": "resnet50",
        "accuracy": float(acc), "precision": float(pr), "recall": float(rc), "f1": float(f1),
        "params_m": round(params_m, 2),
        "inference_ms": round(mean_ms, 3),
        "notes": f"Test set: {len(y_true)} samples",
    })
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CATEGORIES))))
    db.save_confusion_matrix(CATEGORIES, cm.tolist(), model="resnet50")

    print(json.dumps({"model": "resnet50", "accuracy": acc, "f1": f1,
                      "params_m": round(params_m, 2), "inference_ms": round(mean_ms, 3)}, indent=2))


if __name__ == "__main__":
    main()
