"""
Rule-based segregation and alert logic.
NO ML, NO LLM — deterministic dictionary lookups only.
"""
from __future__ import annotations

from typing import List

from config import BIN_MAP, HAZARDOUS_CATEGORIES, CONFIDENCE_THRESHOLD


def assign_bin(category: str) -> str:
    """Map a waste category to a colour-coded bin."""
    return BIN_MAP.get(category, "Black Bin")


def compute_alerts(category: str, confidence: float) -> List[str]:
    """Return all alert tags that apply to a single detection."""
    alerts: List[str] = []
    if confidence < CONFIDENCE_THRESHOLD:
        alerts.append("LOW_CONFIDENCE")
    if category in HAZARDOUS_CATEGORIES:
        alerts.append("HAZARDOUS")
    return alerts
