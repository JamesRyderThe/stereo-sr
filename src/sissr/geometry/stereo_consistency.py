from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor


def disparity_warp(
    image: Tensor,
    disparity: Tensor,
    direction: Literal["right_to_left", "left_to_right"] = "right_to_left",
) -> Tensor:
    B, _, H, W = image.shape
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, H, device=image.device),
        torch.linspace(-1.0, 1.0, W, device=image.device),
        indexing="ij",
    )
    grid_x = grid_x.unsqueeze(0).expand(B, -1, -1)
    grid_y = grid_y.unsqueeze(0).expand(B, -1, -1)

    disp_normalized = disparity.squeeze(1) * 2.0 / (W - 1)
    if direction == "right_to_left":
        sample_x = grid_x + disp_normalized
    else:
        sample_x = grid_x - disp_normalized

    grid = torch.stack([sample_x, grid_y], dim=-1)
    return F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=True)


def lr_consistency_mask(disp_left: Tensor, disp_right: Tensor, threshold: float = 1.0) -> Tensor:
    warped_right = disparity_warp(disp_right, disp_left, direction="right_to_left")
    diff = torch.abs(disp_left - warped_right)
    return (diff < threshold).float()


def photometric_error(left: Tensor, right: Tensor, disp_left: Tensor) -> Tensor:
    warped_right = disparity_warp(right, disp_left, direction="right_to_left")
    return torch.mean(torch.abs(left - warped_right), dim=1, keepdim=True)


def confidence_from_consistency(
    consistency_mask: Tensor, photo_err: Tensor, photo_threshold: float = 0.1
) -> Tensor:
    photo_confidence = 1.0 - torch.clamp(photo_err / photo_threshold, 0.0, 1.0)
    return consistency_mask * photo_confidence
