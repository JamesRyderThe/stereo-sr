from __future__ import annotations

import torch
from torch import Tensor


def charbonnier_loss(pred: Tensor, target: Tensor, eps: float = 1e-12) -> Tensor:
    return torch.sqrt((pred - target) ** 2 + eps**2).mean()


def stereo_charbonnier_loss(sr: Tensor, gt: Tensor, eps: float = 1e-12) -> Tensor:
    return charbonnier_loss(sr[:, :3], gt[:, :3], eps) + charbonnier_loss(
        sr[:, 3:], gt[:, 3:], eps
    )
