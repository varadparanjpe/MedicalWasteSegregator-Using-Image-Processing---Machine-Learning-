"""
Generate a confusion matrix + per-class metrics + inference latency for a trained
classifier (EfficientNet-B0 by default) and persist everything into the FastAPI
SQLite DB so the dashboard can render it.

Also writes a PNG heat-map for reports.

Usage:
    python confusion_matrix.py \
        --data-root /Users/varadparanjpe/Downloads/processed_dataset \
        --weights   ../../backend/weights/efficientnet_b0.pth \
        --model     efficientnet_b0
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score

sys.path.append(str(Path(__file__).resolve().parents[2] / "backend"))
from config import CATEGORIES                                        # noqa: E402
from database import db                                              # noqa: E402
from models.classification_model import build_efficientnet           # noqa: E402
from common import device, eval_transform, loader                    # noqa: E402


@torch.no_grad()
def infer_all(model, dl, dev):
    model.eval()
    y_true, y_pred, y_conf = [], [], []
    times = []
    for x, y in dl:
        x = x.to(dev)
        t0 = time.perf_counter()
        logits = model(x)
        times.append((time.perf_counter() - t0) * 1000 / x.size(0))
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        y_true += y.tolist()
        y_pred += pred.cpu().tolist()
        y_conf += conf.cpu().tolist()
    return y_true, y_pred, y_conf, float(np.mean(times))


def build_model(model_name: str, num_classes: int):
    if model_name == "efficientnet_b0":
        return build_efficientnet(num_classes=num_classes, pretrained=False)
    if model_name == "resnet50":
        from torchvision.models import resnet50
        m = resnet50(weights=None)
        import torch.nn as nn
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    raise ValueError(f"Unknown model {model_name}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--weights",   required=True, type=Path)
    p.add_argument("--model", default="efficientnet_b0",
                   choices=["efficientnet_b0", "resnet50"])
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--out-png", type=Path, default=Path("confusion_matrix.png"))
    args = p.parse_args()

    dev = device()
    dl = loader(args.data_root, "test", batch=args.batch, tf=eval_transform())
    model = build_model(args.model, len(CATEGORIES)).to(dev)
    state = torch.load(args.weights, map_location=dev)
    model.load_state_dict(state if not isinstance(state, dict) or "state_dict" not in state else state["state_dict"], strict=False)

    y_true, y_pred, y_conf, mean_ms = infer_all(model, dl, dev)

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    per_p, per_r, per_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(CATEGORIES))), zero_division=0
    )
    per_class = {
        CATEGORIES[i]: {"precision": float(per_p[i]), "recall": float(per_r[i]), "f1": float(per_f1[i])}
        for i in range(len(CATEGORIES))
    }

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CATEGORIES))))
    print("Accuracy:", acc, "F1:", f1)
    print("Per-class:", json.dumps(per_class, indent=2))
    print("CM:\n", cm)

    # ─── Persist into API DB ────────────────────────────────────────────
    db.init_db()
    db.save_model_metrics({"accuracy": acc, "precision": pr, "recall": rc, "f1": f1,
                           "per_class": per_class})
    db.save_confusion_matrix(CATEGORIES, cm.tolist(), model=args.model)

    # Params (millions) for model_comparison table
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    db.save_comparison_row({
        "model": args.model,
        "accuracy": acc, "precision": pr, "recall": rc, "f1": f1,
        "params_m": round(params_m, 2),
        "inference_ms": round(mean_ms, 3),
        "notes": f"Test set: {len(y_true)} samples",
    })

    # ─── Confidence histogram (calibration) ────────────────────────────
    edges = np.linspace(0, 1, 11)
    counts, _ = np.histogram(y_conf, bins=edges)
    correct_arr = np.array(y_true) == np.array(y_pred)
    correct_per_bin = [
        int(correct_arr[(np.array(y_conf) >= edges[i]) & (np.array(y_conf) < edges[i + 1])].sum())
        for i in range(len(edges) - 1)
    ]
    db.save_confidence_hist(edges.tolist(), counts.tolist(), correct_per_bin)

    # ─── PNG heat-map ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="viridis",
                xticklabels=CATEGORIES, yticklabels=CATEGORIES, ax=ax, cbar=False)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {args.model}  (acc={acc:.3f}, F1={f1:.3f})")
    plt.xticks(rotation=30, ha="right"); plt.tight_layout()
    plt.savefig(args.out_png, dpi=150)
    print("Saved heat-map to", args.out_png)


if __name__ == "__main__":
    main()
