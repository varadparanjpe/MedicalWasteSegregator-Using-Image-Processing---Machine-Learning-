"""
Compute inverse-frequency class weights for cross-entropy from the train split.
Call from train_efficientnet.py as an optional flag.

    python class_weights.py --data-root /path/to/processed_dataset
"""
from __future__ import annotations
import argparse, sys
from collections import Counter
from pathlib import Path

import torch
from torchvision import datasets

sys.path.append(str(Path(__file__).resolve().parents[2] / "backend"))
from config import CATEGORIES                                       # noqa: E402


def compute(data_root: Path) -> torch.Tensor:
    ds = datasets.ImageFolder(data_root / "train")
    counts = Counter(ds.targets)
    total = sum(counts.values())
    weights = [total / (len(CATEGORIES) * counts[i]) for i in range(len(CATEGORIES))]
    return torch.tensor(weights, dtype=torch.float32)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True, type=Path)
    args = p.parse_args()
    w = compute(args.data_root)
    for cat, val in zip(CATEGORIES, w.tolist()):
        print(f"{cat:<25} {val:.4f}")
