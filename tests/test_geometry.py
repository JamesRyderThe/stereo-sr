from __future__ import annotations

import pytest
import torch
from torch import Tensor

from sissr.geometry.stereo_consistency import (
    confidence_from_consistency,
    disparity_warp,
    lr_consistency_mask,
    photometric_error,
)


def test_warp_known_disparity_shifts_correctly() -> None:
    H, W = 32, 64
    shift_px = 5.0
    right = torch.zeros(1, 3, H, W)
    right[:, :, :, 20:30] = 1.0
    expected = torch.zeros(1, 3, H, W)
    expected[:, :, :, 15:25] = 1.0
    disp = torch.full((1, 1, H, W), shift_px)
    warped: Tensor = disparity_warp(right, disp, direction="right_to_left")
    interior = slice(None), slice(None), slice(None), slice(5, -5)
    assert torch.allclose(warped[interior], expected[interior], atol=0.15)


def test_warp_zero_disparity_is_identity() -> None:
    image = torch.randn(1, 3, 32, 64)
    disp = torch.zeros(1, 1, 32, 64)
    warped: Tensor = disparity_warp(image, disp, direction="right_to_left")
    assert torch.allclose(warped, image, atol=1e-4)


def test_warp_left_to_right_reverses_direction() -> None:
    H, W = 32, 64
    shift_px = 5.0
    left = torch.zeros(1, 3, H, W)
    left[:, :, :, 15:25] = 1.0
    expected = torch.zeros(1, 3, H, W)
    expected[:, :, :, 20:30] = 1.0
    disp = torch.full((1, 1, H, W), shift_px)
    warped: Tensor = disparity_warp(left, disp, direction="left_to_right")
    interior = slice(None), slice(None), slice(None), slice(5, -5)
    assert torch.allclose(warped[interior], expected[interior], atol=0.15)


def test_lr_consistency_accepts_consistent() -> None:
    H, W = 32, 64
    disp_left = torch.zeros(1, 1, H, W)
    disp_left[:, :, :, : W // 2] = 3.0
    disp_left[:, :, :, W // 2 :] = 8.0
    disp_right = torch.zeros(1, 1, H, W)
    disp_right[:, :, :, : W // 2] = 3.0
    disp_right[:, :, :, W // 2 :] = 8.0
    mask: Tensor = lr_consistency_mask(disp_left, disp_right, threshold=2.0)
    assert mask.mean() > 0.7


def test_lr_consistency_rejects_inconsistent() -> None:
    H, W = 32, 64
    disp_left = torch.zeros(1, 1, H, W)
    disp_left[:, :, :, : W // 2] = 3.0
    disp_right = torch.full((1, 1, H, W), 15.0)
    mask: Tensor = lr_consistency_mask(disp_left, disp_right, threshold=1.0)
    assert mask.mean() < 0.5


def test_photometric_error_zero_for_identical() -> None:
    image = torch.randn(1, 3, 32, 64)
    disp = torch.zeros(1, 1, 32, 64)
    err: Tensor = photometric_error(image, image, disp)
    assert err.shape == (1, 1, 32, 64)
    assert err.max() < 1e-4


def test_photometric_error_nonzero_for_different() -> None:
    left = torch.ones(1, 3, 32, 64)
    right = torch.zeros(1, 3, 32, 64)
    disp = torch.zeros(1, 1, 32, 64)
    err: Tensor = photometric_error(left, right, disp)
    assert err.mean() == pytest.approx(1.0, abs=0.01)


def test_confidence_combines_signals() -> None:
    mask = torch.ones(1, 1, 32, 64)
    low_err = torch.zeros(1, 1, 32, 64)
    conf: Tensor = confidence_from_consistency(mask, low_err)
    assert conf.mean() == pytest.approx(1.0, abs=0.01)

    high_err = torch.full((1, 1, 32, 64), 0.2)
    conf_low: Tensor = confidence_from_consistency(mask, high_err)
    assert conf_low.mean() == pytest.approx(0.0, abs=0.01)
