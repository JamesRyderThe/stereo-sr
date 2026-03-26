from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from sissr.configs.schema import AugmentConfig
from sissr.data.datasets import StereoSRTestDataset, StereoSRTrainDataset


def _disabled_augment() -> AugmentConfig:
    return AugmentConfig(
        hflip_prob=0.0,
        vflip_prob=0.0,
        channel_shuffle_prob=0.0,
        horizontal_shift_prob=0.0,
    )


def test_train_dataset_shapes(synthetic_dataset_dir: Path) -> None:
    ds = StereoSRTrainDataset(
        str(synthetic_dataset_dir),
        scale=4,
        patch_h=32,
        patch_w=96,
        augment=_disabled_augment(),
    )
    assert len(ds) == 2
    item = ds[0]
    assert item["lr"].shape == (6, 32, 96)
    assert item["gt"].shape == (6, 128, 384)


def test_train_dataset_value_range(synthetic_dataset_dir: Path) -> None:
    ds = StereoSRTrainDataset(
        str(synthetic_dataset_dir),
        scale=4,
        patch_h=32,
        patch_w=96,
        augment=_disabled_augment(),
    )
    item = ds[0]
    assert item["lr"].min() >= 0.0
    assert item["lr"].max() <= 1.0
    assert item["gt"].min() >= 0.0
    assert item["gt"].max() <= 1.0


def test_train_dataset_no_geometry_fills_zeros(tmp_path: Path) -> None:
    hr_dir = tmp_path / "train" / "hr"
    lr_dir = tmp_path / "train" / "lr_x4"
    hr_dir.mkdir(parents=True)
    lr_dir.mkdir(parents=True)

    for suffix in ("_L.png", "_R.png"):
        arr = np.random.randint(0, 256, (128, 384, 3), dtype=np.uint8)
        Image.fromarray(arr).save(hr_dir / f"0001{suffix}")
        arr_lr = np.random.randint(0, 256, (32, 96, 3), dtype=np.uint8)
        Image.fromarray(arr_lr).save(lr_dir / f"0001{suffix}")

    ds = StereoSRTrainDataset(
        str(tmp_path),
        scale=4,
        patch_h=32,
        patch_w=96,
        augment=_disabled_augment(),
        use_geometry=False,
    )
    item = ds[0]
    assert torch.equal(item["disp_left"], torch.zeros(1, 32, 96))
    assert torch.equal(item["disp_right"], torch.zeros(1, 32, 96))


def test_train_dataset_use_geometry_loads(synthetic_dataset_dir: Path) -> None:
    ds = StereoSRTrainDataset(
        str(synthetic_dataset_dir),
        scale=4,
        patch_h=32,
        patch_w=96,
        augment=_disabled_augment(),
        use_geometry=True,
    )
    item = ds[0]
    assert not torch.equal(item["disp_left"], torch.zeros(1, 32, 96))


def test_train_dataset_use_geometry_missing_dir_raises(tmp_path: Path) -> None:
    hr_dir = tmp_path / "train" / "hr"
    lr_dir = tmp_path / "train" / "lr_x4"
    hr_dir.mkdir(parents=True)
    lr_dir.mkdir(parents=True)

    for suffix in ("_L.png", "_R.png"):
        arr = np.random.randint(0, 256, (128, 384, 3), dtype=np.uint8)
        Image.fromarray(arr).save(hr_dir / f"0001{suffix}")
        arr_lr = np.random.randint(0, 256, (32, 96, 3), dtype=np.uint8)
        Image.fromarray(arr_lr).save(lr_dir / f"0001{suffix}")

    with pytest.raises(FileNotFoundError, match="Geometry directory not found"):
        StereoSRTrainDataset(
            str(tmp_path),
            scale=4,
            patch_h=32,
            patch_w=96,
            use_geometry=True,
        )


def test_train_dataset_use_geometry_missing_file_raises(synthetic_dataset_dir: Path) -> None:
    geom_dir = synthetic_dataset_dir / "train" / "partial_geometry"
    geom_dir.mkdir()
    np.savez(
        geom_dir / "0001.npz",
        disp_left=np.zeros((32, 96)),
        disp_right=np.zeros((32, 96)),
        conf_left=np.zeros((32, 96)),
        conf_right=np.zeros((32, 96)),
    )
    with pytest.raises(FileNotFoundError, match="Geometry file not found"):
        StereoSRTrainDataset(
            str(synthetic_dataset_dir),
            scale=4,
            patch_h=32,
            patch_w=96,
            augment=_disabled_augment(),
            use_geometry=True,
            geometry_dir="partial_geometry",
        )


def test_train_dataset_use_geometry_invalid_confidence_raises(synthetic_dataset_dir: Path) -> None:
    geom_dir = synthetic_dataset_dir / "train" / "bad_confidence"
    geom_dir.mkdir()
    for stem in ("0001", "0002"):
        np.savez(
            geom_dir / f"{stem}.npz",
            disp_left=np.zeros((32, 96), dtype=np.float32),
            disp_right=np.zeros((32, 96), dtype=np.float32),
            conf_left=np.full((32, 96), 1.5, dtype=np.float32),
            conf_right=np.zeros((32, 96), dtype=np.float32),
        )
    with pytest.raises(ValueError, match="must lie in \\[0, 1\\]"):
        StereoSRTrainDataset(
            str(synthetic_dataset_dir),
            scale=4,
            patch_h=32,
            patch_w=96,
            augment=_disabled_augment(),
            use_geometry=True,
            geometry_dir="bad_confidence",
        )


def test_train_dataset_use_geometry_integer_dtype_raises(synthetic_dataset_dir: Path) -> None:
    geom_dir = synthetic_dataset_dir / "train" / "bad_dtype"
    geom_dir.mkdir()
    for stem in ("0001", "0002"):
        np.savez(
            geom_dir / f"{stem}.npz",
            disp_left=np.zeros((32, 96), dtype=np.int32),
            disp_right=np.zeros((32, 96), dtype=np.float32),
            conf_left=np.zeros((32, 96), dtype=np.float32),
            conf_right=np.zeros((32, 96), dtype=np.float32),
        )
    with pytest.raises(ValueError, match="dtype .* is not floating"):
        StereoSRTrainDataset(
            str(synthetic_dataset_dir),
            scale=4,
            patch_h=32,
            patch_w=96,
            augment=_disabled_augment(),
            use_geometry=True,
            geometry_dir="bad_dtype",
        )


def test_test_dataset_full_image_shapes(synthetic_dataset_dir: Path) -> None:
    ds = StereoSRTestDataset(str(synthetic_dataset_dir), "flickr1024", scale=4)
    assert len(ds) == 1
    item = ds[0]
    assert item["lr"].shape == (6, 32, 96)
    assert item["gt"].shape == (6, 128, 384)
    assert item["stem"] == "0001"
    assert torch.equal(item["disp_left"], torch.zeros(1, 32, 96))


def test_integration_dataloader_batch(synthetic_dataset_dir: Path) -> None:
    ds = StereoSRTrainDataset(
        str(synthetic_dataset_dir),
        scale=4,
        patch_h=32,
        patch_w=96,
        augment=_disabled_augment(),
    )
    loader = DataLoader(ds, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    assert batch["lr"].shape == (2, 6, 32, 96)
    assert batch["gt"].shape == (2, 6, 128, 384)
    assert batch["disp_left"].shape[0] == 2
    assert batch["disp_right"].shape[0] == 2


def test_empty_dataset_root_raises() -> None:
    with pytest.raises(FileNotFoundError):
        StereoSRTrainDataset("/nonexistent/path", scale=4, patch_h=32, patch_w=96)


def test_train_dataset_use_geometry_shape_mismatch_raises(synthetic_dataset_dir: Path) -> None:
    geom_dir = synthetic_dataset_dir / "train" / "bad_shape"
    geom_dir.mkdir()
    for stem in ("0001", "0002"):
        np.savez(
            geom_dir / f"{stem}.npz",
            disp_left=np.zeros((16, 48), dtype=np.float32),
            disp_right=np.zeros((16, 48), dtype=np.float32),
            conf_left=np.zeros((16, 48), dtype=np.float32),
            conf_right=np.zeros((16, 48), dtype=np.float32),
        )
    with pytest.raises(ValueError, match="shape"):
        StereoSRTrainDataset(
            str(synthetic_dataset_dir),
            scale=4,
            patch_h=32,
            patch_w=96,
            augment=_disabled_augment(),
            use_geometry=True,
            geometry_dir="bad_shape",
        )


def test_train_dataset_use_geometry_nonfinite_raises(synthetic_dataset_dir: Path) -> None:
    geom_dir = synthetic_dataset_dir / "train" / "bad_finite"
    geom_dir.mkdir()
    for stem in ("0001", "0002"):
        np.savez(
            geom_dir / f"{stem}.npz",
            disp_left=np.full((32, 96), float("nan"), dtype=np.float32),
            disp_right=np.zeros((32, 96), dtype=np.float32),
            conf_left=np.zeros((32, 96), dtype=np.float32),
            conf_right=np.zeros((32, 96), dtype=np.float32),
        )
    with pytest.raises(ValueError, match="non-finite"):
        StereoSRTrainDataset(
            str(synthetic_dataset_dir),
            scale=4,
            patch_h=32,
            patch_w=96,
            augment=_disabled_augment(),
            use_geometry=True,
            geometry_dir="bad_finite",
        )


def test_missing_paired_file_raises(tmp_path: Path) -> None:
    hr_dir = tmp_path / "train" / "hr"
    lr_dir = tmp_path / "train" / "lr_x4"
    hr_dir.mkdir(parents=True)
    lr_dir.mkdir(parents=True)

    arr = np.random.randint(0, 256, (128, 384, 3), dtype=np.uint8)
    Image.fromarray(arr).save(hr_dir / "0001_L.png")

    with pytest.raises(FileNotFoundError, match="Missing paired file"):
        StereoSRTrainDataset(str(tmp_path), scale=4, patch_h=32, patch_w=96)
