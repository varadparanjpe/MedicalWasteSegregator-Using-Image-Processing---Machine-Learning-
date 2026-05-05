"""
run_training.py — Self-contained EfficientNet-B0 trainer for Biomedical Waste AI
==================================================================================
Handles everything in one script:
  1. Splits your flat 'train/' folder into train / val / test (70/15/15)
     while keeping augmented variants of a base image in the same split.
  2. Trains EfficientNet-B0 with:
       - ImageNet-pretrained backbone
       - Optional inverse-frequency class weights (--class-weights)
       - AdamW optimiser + Cosine LR schedule
       - Best-by-val-F1 checkpointing
  3. Evaluates on the held-out test set and prints a full report.

Usage (from the folder containing this script):
    # Step 1 – install deps once:
    pip install torch torchvision scikit-learn tqdm

    # Step 2 – run training:
    python run_training.py

    # Optional flags:
    python run_training.py --epochs 50 --batch-size 64 --class-weights --lr 1e-4

Outputs:
    efficientnet_b0.pth   — best model weights (saved next to this script)
    split_dataset/        — the train/val/test folder structure (auto-created)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
CATEGORIES = [
    "sharps_waste",
    "infectious_waste",
    "pathological_waste",
    "plastic_recyclable",
    "pharmaceutical_waste",
    "general_waste",
]

IMG_SIZE   = 224
PIXEL_MEAN = [0.485, 0.456, 0.406]   # ImageNet stats (good default)
PIXEL_STD  = [0.229, 0.224, 0.225]

SCRIPT_DIR  = Path(__file__).resolve().parent
RAW_TRAIN   = SCRIPT_DIR                        # flat per-class folders live here
SPLIT_DIR   = SCRIPT_DIR / "split_dataset"      # will be created
WEIGHTS_OUT = SCRIPT_DIR / "efficientnet_b0.pth"

SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 42


# ──────────────────────────────────────────────────────────────────────────────
# Step 1 – Dataset split
# ──────────────────────────────────────────────────────────────────────────────
def build_splits(raw: Path, dst: Path, seed: int = SEED) -> None:
    """
    Groups augmented variants by base image name, then splits groups 70/15/15
    so no base image leaks across splits.
    """
    if (dst / "train").exists():
        print(f"[split] Split already exists at {dst}. Skipping.\n")
        return

    print(f"[split] Building train/val/test splits from {raw} …")
    random.seed(seed)

    for cat in CATEGORIES:
        src_dir = raw / cat
        if not src_dir.exists():
            sys.exit(f"ERROR: Expected class folder not found: {src_dir}")

        all_files = sorted(src_dir.glob("*.jpg"))
        groups: dict[str, list[Path]] = defaultdict(list)
        for f in all_files:
            stem = f.stem
            base = stem[: stem.rfind("_aug")] if "_aug" in stem else stem
            groups[base].append(f)

        bases = list(groups.keys())
        random.shuffle(bases)
        n = len(bases)
        n_train = int(n * SPLIT_RATIOS["train"])
        n_val   = int(n * SPLIT_RATIOS["val"])

        split_bases = {
            "train": bases[:n_train],
            "val":   bases[n_train : n_train + n_val],
            "test":  bases[n_train + n_val :],
        }

        for split, base_list in split_bases.items():
            out = dst / split / cat
            out.mkdir(parents=True, exist_ok=True)
            for base in base_list:
                for f in groups[base]:
                    shutil.copy2(f, out / f.name)

        counts = {s: sum(len(groups[b]) for b in bl) for s, bl in split_bases.items()}
        print(f"  {cat:<25} total={len(all_files):>6}  "
              f"train={counts['train']:>5}  val={counts['val']:>5}  test={counts['test']:>5}")

    print("[split] Done.\n")


# ──────────────────────────────────────────────────────────────────────────────
# Step 2 – Transforms & loaders
# ──────────────────────────────────────────────────────────────────────────────
def get_transforms(split: str):
    if split == "train":
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(PIXEL_MEAN, PIXEL_STD),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(PIXEL_MEAN, PIXEL_STD),
    ])


def make_loader(split_dir: Path, split: str, batch: int, shuffle: bool) -> DataLoader:
    # Pass CATEGORIES explicitly so ImageFolder uses our order, not alphabetical
    ds = datasets.ImageFolder(split_dir / split, transform=get_transforms(split))
    # Remap class_to_idx to match CATEGORIES order
    ds.class_to_idx = {cls: i for i, cls in enumerate(CATEGORIES)}
    ds.samples = [(path, ds.class_to_idx[ds.classes[label]]) for path, label in ds.samples]
    ds.targets = [s[1] for s in ds.samples]
    workers = min(4, os.cpu_count() or 1)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle,
                      num_workers=workers, pin_memory=torch.cuda.is_available())


# ──────────────────────────────────────────────────────────────────────────────
# Step 3 – Model
# ──────────────────────────────────────────────────────────────────────────────
def build_model(num_classes: int = len(CATEGORIES)) -> nn.Module:
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Step 4 – Evaluation helper
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model: nn.Module, dl: DataLoader, dev: torch.device):
    model.eval()
    y_true, y_pred, y_conf, times = [], [], [], []
    for x, y in tqdm(dl, desc="  eval", leave=False):
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
    pr, rc, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return acc, pr, rc, f1, y_true, y_pred, y_conf, float(np.mean(times))


# ──────────────────────────────────────────────────────────────────────────────
# Step 5 – Class weights
# ──────────────────────────────────────────────────────────────────────────────
def compute_class_weights(split_dir: Path) -> torch.Tensor:
    from collections import Counter
    ds = datasets.ImageFolder(split_dir / "train")
    counts = Counter(ds.targets)
    total = sum(counts.values())
    weights = [total / (len(CATEGORIES) * counts[i]) for i in range(len(CATEGORIES))]
    return torch.tensor(weights, dtype=torch.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────────────────────────────────────
def train(args):
    # ── 1. Splits ──────────────────────────────────────────────────────────
    build_splits(RAW_TRAIN, SPLIT_DIR)

    # ── 2. Device ──────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        dev = torch.device("cuda")
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")   # Apple Silicon GPU
    else:
        dev = torch.device("cpu")
    print(f"[device] Using: {dev}\n")

    # ── 3. Data loaders ────────────────────────────────────────────────────
    tr = make_loader(SPLIT_DIR, "train", args.batch_size, shuffle=True)
    va = make_loader(SPLIT_DIR, "val",   args.batch_size, shuffle=False)
    te = make_loader(SPLIT_DIR, "test",  args.batch_size, shuffle=False)
    print(f"[data] train={len(tr.dataset)}  val={len(va.dataset)}  test={len(te.dataset)}\n")

    # ── 4. Model ───────────────────────────────────────────────────────────
    model = build_model().to(dev)
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[model] EfficientNet-B0  params={params_m:.2f}M\n")

    # ── 5. Loss, optimiser, scheduler ─────────────────────────────────────
    if args.class_weights:
        w = compute_class_weights(SPLIT_DIR).to(dev)
        print("[loss] Inverse-frequency class weights:")
        for cat, val in zip(CATEGORIES, w.tolist()):
            print(f"  {cat:<25} {val:.4f}")
        print()
        loss_fn = nn.CrossEntropyLoss(weight=w)
    else:
        loss_fn = nn.CrossEntropyLoss()

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    # ── 6. Training loop ───────────────────────────────────────────────────
    best_f1 = -1.0
    print("=" * 60)
    print(f"Training for {args.epochs} epochs  (batch={args.batch_size}, lr={args.lr})")
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, n = 0.0, 0
        for x, y in tqdm(tr, desc=f"epoch {epoch:>3}/{args.epochs}"):
            x, y = x.to(dev), y.to(dev)
            optim.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optim.step()
            running_loss += loss.item() * x.size(0)
            n += x.size(0)
        sched.step()

        acc, _, _, f1, *_ = evaluate(model, va, dev)
        train_loss = running_loss / max(n, 1)
        print(f"  epoch {epoch:>3}  loss={train_loss:.4f}  val_acc={acc:.4f}  val_f1={f1:.4f}", end="")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), WEIGHTS_OUT)
            print(f"  ✓ saved (best val_f1={f1:.4f})")
        else:
            print()

    # ── 7. Test-set evaluation ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)
    model.load_state_dict(torch.load(WEIGHTS_OUT, map_location=dev))
    acc, pr, rc, f1, y_true, y_pred, y_conf, mean_ms = evaluate(model, te, dev)

    print(f"\n  Accuracy : {acc:.4f}")
    print(f"  Precision: {pr:.4f}  (macro)")
    print(f"  Recall   : {rc:.4f}  (macro)")
    print(f"  F1       : {f1:.4f}  (macro)")
    print(f"  Latency  : {mean_ms:.2f} ms/image")
    print(f"  Params   : {params_m:.2f} M\n")

    print("Per-class report:")
    print(classification_report(y_true, y_pred, target_names=CATEGORIES, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix (rows=true, cols=pred):")
    header = "".join(f"{c[:6]:>8}" for c in CATEGORIES)
    print(f"{'':>25}{header}")
    for i, row in enumerate(cm):
        row_str = "".join(f"{v:>8}" for v in row)
        print(f"  {CATEGORIES[i]:<23}{row_str}")

    # Save summary JSON alongside weights
    summary = {
        "accuracy": round(float(acc), 4),
        "precision_macro": round(float(pr), 4),
        "recall_macro": round(float(rc), 4),
        "f1_macro": round(float(f1), 4),
        "inference_ms_per_image": round(mean_ms, 3),
        "params_M": round(params_m, 2),
        "test_samples": len(y_true),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "class_weights": args.class_weights,
    }
    summary_path = WEIGHTS_OUT.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"\n[done] Weights saved → {WEIGHTS_OUT}")
    print(f"[done] Summary saved → {summary_path}")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train EfficientNet-B0 on biomedical waste dataset")
    p.add_argument("--epochs",       type=int,   default=30,   help="Number of training epochs (default: 30)")
    p.add_argument("--batch-size",   type=int,   default=32,   help="Batch size (default: 32)")
    p.add_argument("--lr",           type=float, default=3e-4, help="Initial learning rate (default: 3e-4)")
    p.add_argument("--class-weights", action="store_true",     help="Use inverse-frequency class weights")
    args = p.parse_args()
    train(args)
