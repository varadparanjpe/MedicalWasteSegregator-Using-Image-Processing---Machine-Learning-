"""Image utilities: base64 codec, cropping, annotated preview rendering."""
from __future__ import annotations

import base64
import io
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def decode_b64_to_image(b64: str) -> Image.Image:
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    data = base64.b64decode(b64)
    return Image.open(io.BytesIO(data)).convert("RGB")


def encode_image_to_b64(img: Image.Image, fmt: str = "JPEG", quality: int = 90) -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def crop_bbox(img: Image.Image, bbox: Tuple[float, float, float, float]) -> Image.Image:
    x1, y1, x2, y2 = map(int, bbox)
    w, h = img.size
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return img.copy()
    return img.crop((x1, y1, x2, y2))


_COLORS = {
    "sharps_waste":         (231, 76, 60),
    "infectious_waste":     (243, 156, 18),
    "pathological_waste":   (142, 68, 173),
    "plastic_recyclable":   (52, 152, 219),
    "pharmaceutical_waste": (26, 188, 156),
    "general_waste":        (149, 165, 166),
}


def render_annotated(image: Image.Image, detections: List[dict]) -> Image.Image:
    """Draw bounding boxes with category+confidence onto a copy of the image."""
    arr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    for det in detections:
        x1, y1, x2, y2 = map(int, (det["bbox"]["x1"], det["bbox"]["y1"], det["bbox"]["x2"], det["bbox"]["y2"]))
        color = _COLORS.get(det["category"], (200, 200, 200))[::-1]  # BGR
        cv2.rectangle(arr, (x1, y1), (x2, y2), color, 2)
        label = f'{det["category"]} {det["classification_confidence"]:.2f}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(arr, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(arr, label, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
