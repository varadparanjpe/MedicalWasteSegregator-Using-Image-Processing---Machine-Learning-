"""Shared helpers for ML training/eval scripts."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "backend"))
from config import CATEGORIES, IMG_SIZE, PIXEL_MEAN, PIXEL_STD   # noqa: E402


def eval_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(PIXEL_MEAN, PIXEL_STD),
    ])


def train_transform(use_aug: bool = True):
    if use_aug:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize(PIXEL_MEAN, PIXEL_STD),
        ])
    return eval_transform()


def loader(root: Path, split: str, batch: int = 32, tf=None, shuffle: bool = False):
    ds = datasets.ImageFolder(root / split, transform=tf or eval_transform())
    assert ds.classes == CATEGORIES, (
        f"Dataset classes {ds.classes} != config CATEGORIES {CATEGORIES}. "
        "Re-run MLA_PreProcessing.py or align config.CATEGORIES."
    )
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=4, pin_memory=True)


def device(dev: str | None = None) -> torch.device:
    return torch.device(dev or ("cuda" if torch.cuda.is_available() else "cpu"))
