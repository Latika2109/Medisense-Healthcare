"""Module 4 - PneumoniaMNIST CNN and Grad-CAM (Exp 3)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class SmallCNN(nn.Module):
    """3-block CNN for 1x28x28 images -> 1 logit (pneumonia)."""

    def __init__(self) -> None:
        super().__init__()
        def block(i: int, o: int) -> nn.Sequential:
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU())
        self.b1, self.b2, self.b3 = block(1, 16), block(16, 32), block(32, 64)
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.b1(x))
        x = self.pool(self.b2(x))
        x = self.b3(x)  # last conv features (Grad-CAM target)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(self.drop(x)).squeeze(1)


def to_tensor(images: np.ndarray) -> torch.Tensor:
    """uint8 (N,28,28) -> normalised float tensor (N,1,28,28)."""
    t = torch.from_numpy(np.asarray(images, dtype=np.float32) / 255.0).unsqueeze(1)
    return (t - 0.5) / 0.5


def predict_proba(model: nn.Module, images: np.ndarray) -> np.ndarray:
    """Pneumonia probability for each uint8 image."""
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(to_tensor(images))).numpy()


def grad_cam(model: SmallCNN, image: np.ndarray) -> np.ndarray:
    """Grad-CAM heatmap in [0,1], shape (28,28), for the pneumonia logit of one uint8 image."""
    model.eval()
    feats: list[torch.Tensor] = []

    def _save(_m: nn.Module, _i: object, out: torch.Tensor) -> None:
        out.retain_grad()
        feats.append(out)

    h = model.b3.register_forward_hook(_save)
    x = to_tensor(image[None])
    try:
        logit = model(x)[0]
        model.zero_grad()
        logit.backward()
    finally:
        h.remove()
    a = feats[0]
    w = a.grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * a).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)[0, 0].detach().numpy()
    return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
