"""
Train DenseNet-121 on the preprocessed biomedical waste dataset.

Features:
  - Optional inverse-frequency class-weighted cross-entropy (--class-weights)
  - Cosine LR schedule, AdamW, best-by-F1 checkpointing
  - On finish: confusion matrix + per-class metrics + latency → DB

Usage:
    python train_densenet.py \
        --data-root /path/to/processed_dataset \
        --out ../../backend/weights/densenet121.pth \
        --epochs 30 --batch-size 32 --class-weights

Notes:
  - DenseNet-121 has ~7.98M params (B0: 5.3M). Dense skip connections
    between ALL layer pairs reduce vanishing gradient — beneficial for
    small biomedical datasets where features from earlier layers matter.
  - Expected test accuracy: ~91.5-92.5% — competitive with B0 but at
    ~2× inference cost due to memory-bound dense connections.
  - Academic context: DenseNet-121 = backbone of CheXNet (Stanford 2017)
    for chest X-ray pathology detection — well-understood in biomedical imaging.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "backend"))
from config import CATEGORIES                                          # noqa: E402
from database import db                                                # noqa: E402
from models.densenet_model import build_densenet121                    # noqa: E402
from common import device, eval_transform, loader, train_transform     # noqa: E402
from class_weights import compute as compute_class_weights             # noqa: E402


@torch.no_grad()
def evaluate(model, dl, dev):
    model.eval()
    y_true, y_pred, y_conf = [], [], []
    times = []
    for x, y in tqdm(dl, desc="eval", leave=False):
        x = x.to(dev)
        t0 = time.perf_counter()
        logits = model(x)
        times.append((time.perf_counter() - t0) * 1000 / x.size(0))
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        y_true += y.tolist()
        y_pred += pred.cpu().tolist()
        y_conf += conf.cpu().tolist()
    acc = accuracy_score(y_true, y_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    per_p, per_r, per_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(CATEGORIES))), zero_division=0
    )
    per_class = {
        CATEGORIES[i]: {"precision": float(per_p[i]), "recall": float(per_r[i]), "f1": float(per_f1[i])}
        for i in range(len(CATEGORIES))
    }
    return acc, pr, rc, f1, per_class, y_true, y_pred, y_conf, float(np.mean(times))


def main():
    p = argparse.ArgumentParser(description="Train DenseNet-121 for biomedical waste classification")
    p.add_argument("--data-root",    required=True, type=Path)
    p.add_argument("--out",          required=True, type=Path)
    p.add_argument("--epochs",       type=int,   default=30)
    p.add_argument("--batch-size",   type=int,   default=32)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--class-weights", action="store_true",
                   help="Use inverse-frequency weighted cross-entropy")
    args = p.parse_args()

    dev = device()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    tr = loader(args.data_root, "train", args.batch_size, tf=train_transform(True), shuffle=True)
    va = loader(args.data_root, "val",   args.batch_size, tf=eval_transform())
    te = loader(args.data_root, "test",  args.batch_size, tf=eval_transform())

    model = build_densenet121(num_classes=len(CATEGORIES), pretrained=True).to(dev)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    if args.class_weights:
        w = compute_class_weights(args.data_root).to(dev)
        print("Class weights:", {c: round(v, 3) for c, v in zip(CATEGORIES, w.tolist())})
        loss_fn = nn.CrossEntropyLoss(weight=w)
    else:
        loss_fn = nn.CrossEntropyLoss()

    best_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, n = 0.0, 0
        for x, y in tqdm(tr, desc=f"epoch {epoch}/{args.epochs}"):
            x, y = x.to(dev), y.to(dev)
            optim.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optim.step()
            running += loss.item() * x.size(0); n += x.size(0)
        sched.step()

        acc, _, _, f1, *_ = evaluate(model, va, dev)
        print(f"[epoch {epoch}] train_loss={running/max(n,1):.4f} val_acc={acc:.4f} val_f1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), args.out)
            print(f"  ↳ saved best weights (val f1={f1:.4f})")

    # ─── Test-set evaluation + persist to dashboard DB ────────────────────
    model.load_state_dict(torch.load(args.out, map_location=dev))
    acc, pr, rc, f1, per_class, y_true, y_pred, y_conf, mean_ms = evaluate(model, te, dev)

    db.init_db()
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CATEGORIES))))
    db.save_confusion_matrix(CATEGORIES, cm.tolist(), model="densenet121")

    params_m = sum(p_.numel() for p_ in model.parameters()) / 1e6
    db.save_comparison_row({
        "model": "densenet121",
        "accuracy": float(acc), "precision": float(pr), "recall": float(rc), "f1": float(f1),
        "params_m": round(params_m, 2),
        "inference_ms": round(mean_ms, 3),
        "notes": f"Test set: {len(y_true)} samples · CheXNet backbone · biomedical domain",
    })

    edges = np.linspace(0, 1, 11)
    counts, _ = np.histogram(y_conf, bins=edges)
    correct_arr = np.array(y_true) == np.array(y_pred)
    per_bin_correct = [
        int(correct_arr[(np.array(y_conf) >= edges[i]) & (np.array(y_conf) < edges[i+1])].sum())
        for i in range(len(edges) - 1)
    ]
    db.save_confidence_hist(edges.tolist(), counts.tolist(), per_bin_correct)

    print("\nTEST SET RESULTS — DenseNet-121")
    print(json.dumps({"accuracy": acc, "precision": pr, "recall": rc, "f1": f1,
                      "params_m": round(params_m, 2), "inference_ms": round(mean_ms, 3)}, indent=2))


if __name__ == "__main__":
    main()
