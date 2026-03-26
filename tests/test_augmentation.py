from __future__ import annotations

import pytest
import torch
from torch import Tensor

from sissr.configs.schema import AugmentConfig
from sissr.data.augmentation import (
    StereoSample,
    apply_augmentation,
    channel_shuffle,
    horizontal_shift,
    random_crop,
    stereo_hflip,
    stereo_vflip,
)


def _make_sample(lr_h: int = 32, lr_w: int = 64, scale: int = 4) -> StereoSample:
    return StereoSample(
        lr_left=torch.randn(3, lr_h, lr_w),
        lr_right=torch.randn(3, lr_h, lr_w),
        gt_left=torch.randn(3, lr_h * scale, lr_w * scale),
        gt_right=torch.randn(3, lr_h * scale, lr_w * scale),
        disp_left=torch.randn(1, lr_h, lr_w),
        disp_right=torch.randn(1, lr_h, lr_w),
        conf_left=torch.randn(1, lr_h, lr_w),
        conf_right=torch.randn(1, lr_h, lr_w),
    )


def _disabled_augment() -> AugmentConfig:
    return AugmentConfig(
        hflip_prob=0.0,
        vflip_prob=0.0,
        channel_shuffle_prob=0.0,
        horizontal_shift_prob=0.0,
    )


def test_random_crop_shapes() -> None:
    scale = 4
    sample = _make_sample(lr_h=32, lr_w=64, scale=scale)
    cropped = random_crop(sample, patch_h=16, patch_w=24, scale=scale)

    assert cropped["lr_left"].shape == (3, 16, 24)
    assert cropped["lr_right"].shape == (3, 16, 24)
    assert cropped["gt_left"].shape == (3, 64, 96)
    assert cropped["gt_right"].shape == (3, 64, 96)
    assert cropped["disp_left"].shape == (1, 16, 24)
    assert cropped["disp_right"].shape == (1, 16, 24)
    assert cropped["conf_left"].shape == (1, 16, 24)
    assert cropped["conf_right"].shape == (1, 16, 24)


def test_hflip_swaps_lr() -> None:
    sample = _make_sample()
    result = stereo_hflip(sample)

    assert torch.equal(result["lr_left"], sample["lr_right"].flip(-1))
    assert torch.equal(result["lr_right"], sample["lr_left"].flip(-1))
    assert torch.equal(result["gt_left"], sample["gt_right"].flip(-1))
    assert torch.equal(result["gt_right"], sample["gt_left"].flip(-1))


def test_hflip_geometry_invariant() -> None:
    sample = _make_sample()
    result = stereo_hflip(sample)

    assert torch.equal(result["disp_left"], sample["disp_right"].flip(-1))
    assert torch.equal(result["disp_right"], sample["disp_left"].flip(-1))
    assert torch.equal(result["conf_left"], sample["conf_right"].flip(-1))
    assert torch.equal(result["conf_right"], sample["conf_left"].flip(-1))


def test_vflip_preserves_lr_identity() -> None:
    sample = _make_sample()
    result = stereo_vflip(sample)

    assert torch.equal(result["lr_left"], sample["lr_left"].flip(-2))
    assert torch.equal(result["lr_right"], sample["lr_right"].flip(-2))
    assert torch.equal(result["gt_left"], sample["gt_left"].flip(-2))
    assert torch.equal(result["gt_right"], sample["gt_right"].flip(-2))


def test_channel_shuffle_same_permutation() -> None:
    sample = _make_sample()
    torch.manual_seed(42)
    result = channel_shuffle(sample)

    lr_left_orig = sample["lr_left"]
    lr_left_out = result["lr_left"]
    perm: list[int] = []
    for c in range(3):
        for p in range(3):
            if torch.equal(lr_left_out[c], lr_left_orig[p]):
                perm.append(p)
                break

    assert len(perm) == 3
    perm_t = torch.tensor(perm)

    for key in ("lr_right", "gt_left", "gt_right"):
        expected: Tensor = sample[key][perm_t]  # type: ignore[literal-required]
        assert torch.equal(result[key], expected)  # type: ignore[literal-required]


def test_channel_shuffle_no_geometry_change() -> None:
    sample = _make_sample()
    result = channel_shuffle(sample)

    assert torch.equal(result["disp_left"], sample["disp_left"])
    assert torch.equal(result["disp_right"], sample["disp_right"])
    assert torch.equal(result["conf_left"], sample["conf_left"])
    assert torch.equal(result["conf_right"], sample["conf_right"])


def test_horizontal_shift_scales_gt_offset() -> None:
    sample = StereoSample(
        lr_left=torch.tensor([[[0.0, 1.0, 2.0, 3.0]]] * 3),
        lr_right=torch.tensor([[[4.0, 5.0, 6.0, 7.0]]] * 3),
        gt_left=torch.tensor([[[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]]] * 3),
        gt_right=torch.tensor([[[8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]]] * 3),
        disp_left=torch.tensor([[[0.0, 1.0, 2.0, 3.0]]]),
        disp_right=torch.tensor([[[4.0, 5.0, 6.0, 7.0]]]),
        conf_left=torch.tensor([[[0.0, 1.0, 2.0, 3.0]]]),
        conf_right=torch.tensor([[[4.0, 5.0, 6.0, 7.0]]]),
    )

    result = horizontal_shift(sample, 1)

    assert torch.equal(result["lr_left"][0], torch.tensor([[1.0, 0.0, 1.0, 2.0]]))
    assert torch.equal(
        result["gt_left"][0],
        torch.tensor([[2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]]),
    )
    assert torch.equal(result["disp_left"], torch.tensor([[[1.0, 0.0, 1.0, 2.0]]]))


def test_apply_augmentation_can_trigger_horizontal_shift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample = StereoSample(
        lr_left=torch.tensor([[[0.0, 1.0, 2.0, 3.0]]] * 3),
        lr_right=torch.tensor([[[4.0, 5.0, 6.0, 7.0]]] * 3),
        gt_left=torch.tensor([[[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]]] * 3),
        gt_right=torch.tensor([[[8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]]] * 3),
        disp_left=torch.tensor([[[0.0, 1.0, 2.0, 3.0]]]),
        disp_right=torch.tensor([[[4.0, 5.0, 6.0, 7.0]]]),
        conf_left=torch.tensor([[[0.0, 1.0, 2.0, 3.0]]]),
        conf_right=torch.tensor([[[4.0, 5.0, 6.0, 7.0]]]),
    )
    augment = AugmentConfig(
        hflip_prob=0.0,
        vflip_prob=0.0,
        channel_shuffle_prob=0.0,
        horizontal_shift_prob=1.0,
        horizontal_shift_max_px=1,
    )
    rand_values = iter(
        [
            torch.tensor([0.0]),
            torch.tensor([1.0]),
            torch.tensor([1.0]),
            torch.tensor([1.0]),
        ]
    )
    monkeypatch.setattr(
        "sissr.data.augmentation.torch.rand", lambda *args, **kwargs: next(rand_values)
    )
    monkeypatch.setattr(
        "sissr.data.augmentation.torch.randint",
        lambda low, high, size: torch.tensor([1]),
    )

    result = apply_augmentation(sample, augment)

    assert torch.equal(result["lr_left"][0], torch.tensor([[1.0, 0.0, 1.0, 2.0]]))
    assert torch.equal(
        result["gt_left"][0],
        torch.tensor([[2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]]),
    )


def test_augmentation_disabled_is_identity() -> None:
    sample = _make_sample()
    result = apply_augmentation(sample, _disabled_augment())

    for key in sample:
        assert torch.equal(result[key], sample[key])  # type: ignore[literal-required]


def test_double_hflip_is_identity() -> None:
    sample = _make_sample()
    result = stereo_hflip(stereo_hflip(sample))

    for key in sample:
        assert torch.equal(result[key], sample[key])  # type: ignore[literal-required]


def test_random_crop_undersized_image_raises() -> None:
    sample = _make_sample(lr_h=8, lr_w=16, scale=4)
    with pytest.raises(ValueError, match="smaller than patch size"):
        random_crop(sample, patch_h=32, patch_w=96, scale=4)
