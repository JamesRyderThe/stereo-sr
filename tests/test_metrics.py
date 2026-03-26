from __future__ import annotations

from math import exp

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from sissr.eval.metrics import compute_psnr, compute_ssim, compute_stereo_metrics


def test_psnr_identical_is_inf() -> None:
    img = torch.rand(3, 32, 32)
    assert compute_psnr(img, img) == float("inf")


def test_psnr_known_value() -> None:
    a = torch.zeros(3, 32, 32)
    b = torch.ones(3, 32, 32) * 0.1
    result = compute_psnr(a, b)
    assert result == pytest.approx(20.0, abs=1e-4)


def test_psnr_boundary_values() -> None:
    zeros = torch.zeros(3, 32, 32)
    ones = torch.ones(3, 32, 32)
    assert compute_psnr(zeros, zeros) == float("inf")
    assert compute_psnr(ones, ones) == float("inf")
    psnr = compute_psnr(zeros, ones)
    assert psnr == pytest.approx(0.0, abs=1e-4)


def test_ssim_identical_is_one() -> None:
    img = torch.rand(3, 16, 16)
    assert compute_ssim(img, img) == pytest.approx(1.0, abs=1e-6)


def test_ssim_symmetry() -> None:
    a = torch.rand(3, 16, 16)
    b = torch.rand(3, 16, 16)
    assert compute_ssim(a, b) == pytest.approx(compute_ssim(b, a), abs=1e-7)


def test_ssim_valid_conv_shrinks_dims() -> None:
    img_7 = torch.rand(3, 7, 7)
    ssim_7 = compute_ssim(img_7, img_7)
    assert ssim_7 == pytest.approx(1.0, abs=1e-6)

    img_8 = torch.rand(3, 8, 8)
    ssim_8 = compute_ssim(img_8, img_8)
    assert ssim_8 == pytest.approx(1.0, abs=1e-6)

    channels = 3
    kernel = torch.ones(channels, 1, 7, 7) / 49.0
    out_7 = F.conv2d(img_7.unsqueeze(0), kernel, groups=channels)
    out_8 = F.conv2d(img_8.unsqueeze(0), kernel, groups=channels)
    assert out_7.shape == (1, 3, 1, 1)
    assert out_8.shape == (1, 3, 2, 2)


def test_ssim_per_channel_average() -> None:
    r_pred = torch.rand(1, 10, 10)
    r_target = r_pred.clone()

    g_pred = torch.rand(1, 10, 10)
    g_target = torch.rand(1, 10, 10)

    b_pred = torch.rand(1, 10, 10)
    b_target = torch.rand(1, 10, 10)

    pred = torch.cat([r_pred, g_pred, b_pred], dim=0)
    target = torch.cat([r_target, g_target, b_target], dim=0)

    full_ssim = compute_ssim(pred, target)

    ssim_r = compute_ssim(r_pred, r_target)
    ssim_g = compute_ssim(g_pred, g_target)
    ssim_b = compute_ssim(b_pred, b_target)

    expected = (ssim_r + ssim_g + ssim_b) / 3.0
    assert full_ssim == pytest.approx(expected, abs=1e-6)


def test_ssim_hand_computed_numpy_reference() -> None:
    rng = np.random.RandomState(42)
    x_np = rng.rand(3, 7, 7).astype(np.float32)
    y_np = rng.rand(3, 7, 7).astype(np.float32)

    c1 = 1e-4
    c2 = 9e-4
    per_channel_ssim = []
    for ch in range(3):
        mu_x = x_np[ch].mean()
        mu_y = y_np[ch].mean()
        sigma_x_sq = (x_np[ch] ** 2).mean() - mu_x**2
        sigma_y_sq = (y_np[ch] ** 2).mean() - mu_y**2
        sigma_xy = (x_np[ch] * y_np[ch]).mean() - mu_x * mu_y
        num = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
        den = (mu_x**2 + mu_y**2 + c1) * (sigma_x_sq + sigma_y_sq + c2)
        per_channel_ssim.append(num / den)
    expected = float(np.mean(per_channel_ssim))

    x_t = torch.from_numpy(x_np)
    y_t = torch.from_numpy(y_np)
    result = compute_ssim(x_t, y_t)

    assert result == pytest.approx(expected, abs=1e-5)


def test_ssim_gaussian_11x11_gives_different_result() -> None:
    torch.manual_seed(123)
    pred = torch.rand(3, 32, 32)
    target = torch.rand(3, 32, 32)

    our_ssim = compute_ssim(pred, target)

    sigma = 1.5
    coords = torch.arange(11, dtype=torch.float32) - 5.0
    g1d = torch.tensor([exp(-(float(x) ** 2) / (2.0 * sigma**2)) for x in coords])
    g2d = g1d.unsqueeze(1) * g1d.unsqueeze(0)
    g2d = g2d / g2d.sum()
    channels = 3
    gauss_kernel = g2d.unsqueeze(0).unsqueeze(0).expand(channels, 1, 11, 11).contiguous()

    c1 = 1e-4
    c2 = 9e-4
    pred_4d = pred.unsqueeze(0)
    target_4d = target.unsqueeze(0)
    mu_x = F.conv2d(pred_4d, gauss_kernel, groups=channels)
    mu_y = F.conv2d(target_4d, gauss_kernel, groups=channels)
    sigma_x_sq = F.conv2d(pred_4d * pred_4d, gauss_kernel, groups=channels) - mu_x * mu_x
    sigma_y_sq = F.conv2d(target_4d * target_4d, gauss_kernel, groups=channels) - mu_y * mu_y
    sigma_xy = F.conv2d(pred_4d * target_4d, gauss_kernel, groups=channels) - mu_x * mu_y
    num = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    den = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x_sq + sigma_y_sq + c2)
    gauss_ssim_map = num / den
    gauss_ssim = gauss_ssim_map.squeeze(0).mean(dim=(1, 2)).mean().item()

    assert our_ssim != pytest.approx(gauss_ssim, abs=1e-3)


def test_ssim_rejects_wrong_dimensions() -> None:
    with pytest.raises(ValueError, match="Expected \\[C, H, W\\]"):
        compute_ssim(torch.rand(1, 3, 16, 16), torch.rand(1, 3, 16, 16))


def test_ssim_rejects_image_smaller_than_kernel() -> None:
    with pytest.raises(ValueError, match="must be >= 7"):
        compute_ssim(torch.rand(3, 5, 5), torch.rand(3, 5, 5))


def test_stereo_metrics_rejects_wrong_channels() -> None:
    with pytest.raises(ValueError, match="Expected \\[6, H, W\\]"):
        compute_stereo_metrics(torch.rand(3, 32, 32), torch.rand(3, 32, 32))


def test_stereo_metrics_keys_and_averaging() -> None:
    torch.manual_seed(0)
    sr = torch.rand(6, 32, 32)
    gt = torch.rand(6, 32, 32)
    result = compute_stereo_metrics(sr, gt)

    assert set(result.keys()) == {
        "psnr_left",
        "psnr_right",
        "psnr_stereo",
        "ssim_left",
        "ssim_right",
        "ssim_stereo",
    }

    assert result["psnr_stereo"] == pytest.approx(
        (result["psnr_left"] + result["psnr_right"]) / 2.0, abs=1e-7
    )
    assert result["ssim_stereo"] == pytest.approx(
        (result["ssim_left"] + result["ssim_right"]) / 2.0, abs=1e-7
    )
