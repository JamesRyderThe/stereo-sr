from __future__ import annotations

from math import log10
from typing import TypedDict

import torch
import torch.nn.functional as F
from torch import Tensor

_C1 = 1e-4
_C2 = 9e-4


class StereoMetrics(TypedDict):
    psnr_left: float
    psnr_right: float
    psnr_stereo: float
    ssim_left: float
    ssim_right: float
    ssim_stereo: float


def compute_psnr(pred: Tensor, target: Tensor) -> float:
    mse = ((pred - target) ** 2).mean().item()
    if mse == 0.0:
        return float("inf")
    return 10.0 * log10(1.0 / mse)


def compute_ssim(pred: Tensor, target: Tensor) -> float:
    if pred.ndim != 3:
        raise ValueError(f"Expected [C, H, W] input, got {pred.ndim}D")
    if pred.shape[1] < 7 or pred.shape[2] < 7:
        raise ValueError(
            f"Spatial dims ({pred.shape[1]}x{pred.shape[2]}) must be >= 7 for 7x7 SSIM kernel"
        )

    c = pred.shape[0]
    kernel = torch.ones(c, 1, 7, 7, device=pred.device, dtype=pred.dtype) / 49.0
    p = pred.unsqueeze(0)
    t = target.unsqueeze(0)

    mu_p = F.conv2d(p, kernel, groups=c)
    mu_t = F.conv2d(t, kernel, groups=c)
    mu_p_sq = mu_p * mu_p
    mu_t_sq = mu_t * mu_t
    mu_pt = mu_p * mu_t

    sigma_p_sq = F.conv2d(p * p, kernel, groups=c) - mu_p_sq
    sigma_t_sq = F.conv2d(t * t, kernel, groups=c) - mu_t_sq
    sigma_pt = F.conv2d(p * t, kernel, groups=c) - mu_pt

    num = (2.0 * mu_pt + _C1) * (2.0 * sigma_pt + _C2)
    den = (mu_p_sq + mu_t_sq + _C1) * (sigma_p_sq + sigma_t_sq + _C2)

    per_channel = (num / den).squeeze(0).mean(dim=(1, 2))
    return per_channel.mean().item()


def compute_stereo_metrics(sr: Tensor, gt: Tensor) -> StereoMetrics:
    if sr.ndim != 3 or sr.shape[0] != 6:
        raise ValueError(f"Expected [6, H, W] stereo pair, got {list(sr.shape)}")

    psnr_l = compute_psnr(sr[:3], gt[:3])
    psnr_r = compute_psnr(sr[3:], gt[3:])
    ssim_l = compute_ssim(sr[:3], gt[:3])
    ssim_r = compute_ssim(sr[3:], gt[3:])

    return StereoMetrics(
        psnr_left=psnr_l,
        psnr_right=psnr_r,
        psnr_stereo=(psnr_l + psnr_r) / 2.0,
        ssim_left=ssim_l,
        ssim_right=ssim_r,
        ssim_stereo=(ssim_l + ssim_r) / 2.0,
    )
