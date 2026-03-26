from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple, TypedDict

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, decode_image

from sissr.configs.schema import AugmentConfig
from sissr.data.augmentation import StereoSample, apply_augmentation, random_crop

log = logging.getLogger(__name__)


class StereoSRBatch(TypedDict):
    lr: Tensor
    gt: Tensor
    disp_left: Tensor
    disp_right: Tensor
    conf_left: Tensor
    conf_right: Tensor


class StereoSRTestBatch(StereoSRBatch):
    stem: str


class _StereoPaths(NamedTuple):
    hr_left: Path
    hr_right: Path
    lr_left: Path
    lr_right: Path
    geometry: Path | None


class _CachedPair(NamedTuple):
    hr_left: Tensor
    hr_right: Tensor
    lr_left: Tensor
    lr_right: Tensor
    disp_left: Tensor
    disp_right: Tensor
    conf_left: Tensor
    conf_right: Tensor


def _load_uint8(path: Path) -> Tensor:
    image: Tensor = decode_image(str(path), mode=ImageReadMode.RGB)
    return image


def _load_geometry(path: Path, h: int, w: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if not path.exists():
        raise FileNotFoundError(f"Geometry file not found: {path}")
    data = np.load(path)

    tensors: list[Tensor] = []
    for name in ("disp_left", "disp_right", "conf_left", "conf_right"):
        arr = data[name]
        if not np.issubdtype(arr.dtype, np.floating):
            raise ValueError(f"Geometry '{name}' dtype {arr.dtype} is not floating in {path}")
        tensor = torch.from_numpy(arr.copy()).float().unsqueeze(0)
        if tensor.shape != (1, h, w):
            raise ValueError(
                f"Geometry '{name}' shape {tuple(tensor.shape)} != expected "
                f"(1, {h}, {w}) in {path}"
            )
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Geometry '{name}' contains non-finite values in {path}")
        if name.startswith("conf") and (torch.any(tensor < 0.0) or torch.any(tensor > 1.0)):
            raise ValueError(f"Geometry '{name}' must lie in [0, 1] in {path}")
        tensors.append(tensor)

    return tensors[0], tensors[1], tensors[2], tensors[3]


def _zeros_geometry(h: int, w: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return torch.zeros(1, h, w), torch.zeros(1, h, w), torch.zeros(1, h, w), torch.zeros(1, h, w)


def _validate_shapes(
    lr_left: Tensor, lr_right: Tensor, gt_left: Tensor, gt_right: Tensor, scale: int, name: str
) -> None:
    if lr_left.shape != lr_right.shape:
        raise ValueError(f"LR L {tuple(lr_left.shape)} != R {tuple(lr_right.shape)} for {name}")
    if gt_left.shape != gt_right.shape:
        raise ValueError(f"GT L {tuple(gt_left.shape)} != R {tuple(gt_right.shape)} for {name}")
    _, lr_h, lr_w = lr_left.shape
    _, gt_h, gt_w = gt_left.shape
    if gt_h != lr_h * scale or gt_w != lr_w * scale:
        raise ValueError(f"GT ({gt_h}x{gt_w}) != LR ({lr_h}x{lr_w}) * scale ({scale}) for {name}")


def _scan_stereo_pairs(hr_dir: Path, lr_dir: Path, geom_dir: Path | None) -> list[_StereoPaths]:
    if not hr_dir.is_dir():
        raise FileNotFoundError(f"HR directory not found: {hr_dir}")
    left_files = sorted(hr_dir.glob("*_L.png"))
    if not left_files:
        raise FileNotFoundError(f"No stereo pairs found in {hr_dir}")

    pairs: list[_StereoPaths] = []
    for left_path in left_files:
        stem = left_path.name.removesuffix("_L.png")
        right_path = hr_dir / f"{stem}_R.png"
        lr_left_path = lr_dir / f"{stem}_L.png"
        lr_right_path = lr_dir / f"{stem}_R.png"
        for p in (right_path, lr_left_path, lr_right_path):
            if not p.exists():
                raise FileNotFoundError(f"Missing paired file: {p}")
        geom_path = geom_dir / f"{stem}.npz" if geom_dir is not None else None
        pairs.append(_StereoPaths(left_path, right_path, lr_left_path, lr_right_path, geom_path))

    return pairs


def _build_cache(pairs: list[_StereoPaths], scale: int, use_geometry: bool) -> list[_CachedPair]:
    import os

    is_main = int(os.environ.get("LOCAL_RANK", "0")) == 0
    cache: list[_CachedPair] = []
    for i, paths in enumerate(pairs):
        hr_left = _load_uint8(paths.hr_left)
        hr_right = _load_uint8(paths.hr_right)
        lr_left = _load_uint8(paths.lr_left)
        lr_right = _load_uint8(paths.lr_right)
        _validate_shapes(lr_left, lr_right, hr_left, hr_right, scale, paths.hr_left.name)

        _, lr_h, lr_w = lr_left.shape
        if use_geometry and paths.geometry is not None:
            disp_left, disp_right, conf_left, conf_right = _load_geometry(
                paths.geometry, lr_h, lr_w
            )
        else:
            disp_left, disp_right, conf_left, conf_right = _zeros_geometry(lr_h, lr_w)

        cache.append(
            _CachedPair(
                hr_left, hr_right, lr_left, lr_right, disp_left, disp_right, conf_left, conf_right
            )
        )
        if is_main and (i + 1) % 100 == 0:
            log.info("Cached %d/%d pairs", i + 1, len(pairs))

    if is_main:
        log.info("Cached %d pairs in memory", len(cache))
    return cache


class StereoSRTrainDataset(Dataset[StereoSRBatch]):
    def __init__(
        self,
        root: str,
        scale: int,
        patch_h: int,
        patch_w: int,
        *,
        augment: AugmentConfig | None = None,
        use_geometry: bool = False,
        geometry_dir: str = "geometry",
    ) -> None:
        root_path = Path(root)
        hr_dir = root_path / "train" / "hr"
        lr_dir = root_path / "train" / f"lr_x{scale}"

        if use_geometry:
            geom_dir = root_path / "train" / geometry_dir
            if not geom_dir.is_dir():
                raise FileNotFoundError(f"Geometry directory not found: {geom_dir}")
        else:
            geom_dir = None

        pairs = _scan_stereo_pairs(hr_dir, lr_dir, geom_dir)
        self._cache = _build_cache(pairs, scale, use_geometry)
        self.scale = scale
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.augment = augment or AugmentConfig()

    def __len__(self) -> int:
        return len(self._cache)

    def __getitem__(self, idx: int) -> StereoSRBatch:
        cached = self._cache[idx]

        sample = StereoSample(
            lr_left=cached.lr_left,
            lr_right=cached.lr_right,
            gt_left=cached.hr_left,
            gt_right=cached.hr_right,
            disp_left=cached.disp_left,
            disp_right=cached.disp_right,
            conf_left=cached.conf_left,
            conf_right=cached.conf_right,
        )

        sample = random_crop(sample, self.patch_h, self.patch_w, self.scale)
        sample = apply_augmentation(sample, self.augment)

        lr = torch.cat([sample["lr_left"], sample["lr_right"]], dim=0).float() / 255.0
        gt = torch.cat([sample["gt_left"], sample["gt_right"]], dim=0).float() / 255.0

        return StereoSRBatch(
            lr=lr,
            gt=gt,
            disp_left=sample["disp_left"],
            disp_right=sample["disp_right"],
            conf_left=sample["conf_left"],
            conf_right=sample["conf_right"],
        )


class StereoSRTestDataset(Dataset[StereoSRTestBatch]):
    def __init__(
        self,
        root: str,
        dataset_name: str,
        scale: int,
        *,
        use_geometry: bool = False,
        geometry_dir: str = "geometry",
    ) -> None:
        root_path = Path(root)
        hr_dir = root_path / "test" / dataset_name / "hr"
        lr_dir = root_path / "test" / dataset_name / f"lr_x{scale}"

        if use_geometry:
            geom_dir = root_path / "test" / dataset_name / geometry_dir
            if not geom_dir.is_dir():
                raise FileNotFoundError(f"Geometry directory not found: {geom_dir}")
        else:
            geom_dir = None

        pairs = _scan_stereo_pairs(hr_dir, lr_dir, geom_dir)
        self._cache = _build_cache(pairs, scale, use_geometry)
        self._stems = [p.hr_left.name.removesuffix("_L.png") for p in pairs]

    def __len__(self) -> int:
        return len(self._cache)

    def __getitem__(self, idx: int) -> StereoSRTestBatch:
        cached = self._cache[idx]

        lr = torch.cat([cached.lr_left, cached.lr_right], dim=0).float() / 255.0
        gt = torch.cat([cached.hr_left, cached.hr_right], dim=0).float() / 255.0

        return StereoSRTestBatch(
            lr=lr,
            gt=gt,
            disp_left=cached.disp_left,
            disp_right=cached.disp_right,
            conf_left=cached.conf_left,
            conf_right=cached.conf_right,
            stem=self._stems[idx],
        )
