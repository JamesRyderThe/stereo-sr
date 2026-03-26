from __future__ import annotations

from typing import TypedDict

import torch
import torch.nn.functional as F
from torch import Tensor

from sissr.configs.schema import AugmentConfig


class StereoSample(TypedDict):
    lr_left: Tensor
    lr_right: Tensor
    gt_left: Tensor
    gt_right: Tensor
    disp_left: Tensor
    disp_right: Tensor
    conf_left: Tensor
    conf_right: Tensor


def random_crop(sample: StereoSample, patch_h: int, patch_w: int, scale: int) -> StereoSample:
    _, h, w = sample["lr_left"].shape
    if h < patch_h or w < patch_w:
        raise ValueError(f"LR image ({h}x{w}) is smaller than patch size ({patch_h}x{patch_w})")
    top = int(torch.randint(0, h - patch_h + 1, (1,)).item())
    left = int(torch.randint(0, w - patch_w + 1, (1,)).item())

    lr_slice = (..., slice(top, top + patch_h), slice(left, left + patch_w))
    gt_slice = (
        ...,
        slice(top * scale, (top + patch_h) * scale),
        slice(left * scale, (left + patch_w) * scale),
    )

    return StereoSample(
        lr_left=sample["lr_left"][lr_slice],
        lr_right=sample["lr_right"][lr_slice],
        gt_left=sample["gt_left"][gt_slice],
        gt_right=sample["gt_right"][gt_slice],
        disp_left=sample["disp_left"][lr_slice],
        disp_right=sample["disp_right"][lr_slice],
        conf_left=sample["conf_left"][lr_slice],
        conf_right=sample["conf_right"][lr_slice],
    )


def stereo_hflip(sample: StereoSample) -> StereoSample:
    return StereoSample(
        lr_left=sample["lr_right"].flip(-1),
        lr_right=sample["lr_left"].flip(-1),
        gt_left=sample["gt_right"].flip(-1),
        gt_right=sample["gt_left"].flip(-1),
        disp_left=sample["disp_right"].flip(-1),
        disp_right=sample["disp_left"].flip(-1),
        conf_left=sample["conf_right"].flip(-1),
        conf_right=sample["conf_left"].flip(-1),
    )


def stereo_vflip(sample: StereoSample) -> StereoSample:
    return StereoSample(
        lr_left=sample["lr_left"].flip(-2),
        lr_right=sample["lr_right"].flip(-2),
        gt_left=sample["gt_left"].flip(-2),
        gt_right=sample["gt_right"].flip(-2),
        disp_left=sample["disp_left"].flip(-2),
        disp_right=sample["disp_right"].flip(-2),
        conf_left=sample["conf_left"].flip(-2),
        conf_right=sample["conf_right"].flip(-2),
    )


def channel_shuffle(sample: StereoSample) -> StereoSample:
    perm = torch.randperm(3)
    return StereoSample(
        lr_left=sample["lr_left"][perm],
        lr_right=sample["lr_right"][perm],
        gt_left=sample["gt_left"][perm],
        gt_right=sample["gt_right"][perm],
        disp_left=sample["disp_left"],
        disp_right=sample["disp_right"],
        conf_left=sample["conf_left"],
        conf_right=sample["conf_right"],
    )


def horizontal_shift(sample: StereoSample, shift: int) -> StereoSample:
    if shift == 0:
        return sample
    gt_scale = sample["gt_left"].shape[-1] // sample["lr_left"].shape[-1]
    return StereoSample(
        lr_left=_shift_tensor(sample["lr_left"], shift),
        lr_right=_shift_tensor(sample["lr_right"], shift),
        gt_left=_shift_tensor(sample["gt_left"], shift * gt_scale),
        gt_right=_shift_tensor(sample["gt_right"], shift * gt_scale),
        disp_left=_shift_tensor(sample["disp_left"], shift),
        disp_right=_shift_tensor(sample["disp_right"], shift),
        conf_left=_shift_tensor(sample["conf_left"], shift),
        conf_right=_shift_tensor(sample["conf_right"], shift),
    )


def _shift_tensor(tensor: Tensor, shift: int) -> Tensor:
    if shift == 0:
        return tensor
    pad = abs(shift)
    if shift > 0:
        padded = F.pad(tensor, (pad, 0, 0, 0), mode="reflect")
        return padded[..., : tensor.shape[-1]]
    padded = F.pad(tensor, (0, pad, 0, 0), mode="reflect")
    return padded[..., pad:]


def apply_augmentation(
    sample: StereoSample,
    augment: AugmentConfig,
) -> StereoSample:
    max_shift = min(augment.horizontal_shift_max_px, sample["lr_left"].shape[-1] - 1)
    if max_shift > 0 and torch.rand(1).item() < augment.horizontal_shift_prob:
        shift = int(
            torch.randint(
                -max_shift,
                max_shift + 1,
                (1,),
            ).item()
        )
        sample = horizontal_shift(sample, shift)
    if torch.rand(1).item() < augment.hflip_prob:
        sample = stereo_hflip(sample)
    if torch.rand(1).item() < augment.vflip_prob:
        sample = stereo_vflip(sample)
    if torch.rand(1).item() < augment.channel_shuffle_prob:
        sample = channel_shuffle(sample)
    return sample
