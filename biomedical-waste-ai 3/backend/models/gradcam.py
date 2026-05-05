"""
Grad-CAM for EfficientNet-B0.

Attaches to the last convolutional feature layer and returns a heat-map
(numpy array in [0,1]) at the original crop resolution. Callers can overlay
the heat-map onto the image to visualise which regions drove the prediction.
"""
from __future__ import annotations

from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    """
    Usage:
        cam = GradCAM(model, target_layer=model.features[-1])
        heat = cam(input_tensor, class_idx)   # (H, W) np.float32 in [0, 1]
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients:   torch.Tensor | None = None
        self._fh = target_layer.register_forward_hook(self._fwd_hook)
        self._bh = target_layer.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, _m, _i, o):   self.activations = o.detach()
    def _bwd_hook(self, _m, _gi, go):  self.gradients   = go[0].detach()

    def __call__(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad()
        out = self.model(input_tensor)
        out[0, class_idx].backward(retain_graph=True)
        assert self.activations is not None and self.gradients is not None

        # Global-average-pooled gradient weights
        pooled = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (pooled * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
        return cam

    def close(self):
        self._fh.remove()
        self._bh.remove()


def overlay_heatmap(pil: Image.Image, heat: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Render a red→yellow heatmap alpha-blended on the input image."""
    import cv2
    arr = np.array(pil.convert("RGB"))
    H, W = arr.shape[:2]
    heat_u8 = (np.clip(heat, 0, 1) * 255).astype(np.uint8)
    heat_u8 = cv2.resize(heat_u8, (W, H), interpolation=cv2.INTER_LINEAR)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
    blended = (alpha * heat_color + (1 - alpha) * arr).astype(np.uint8)
    return Image.fromarray(blended)
