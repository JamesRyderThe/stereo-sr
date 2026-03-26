from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def _save_rgb(path: Path, h: int, w: int) -> None:
    arr = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)


@pytest.fixture()
def synthetic_dataset_dir(tmp_path: Path) -> Path:
    hr_h, hr_w = 128, 384
    lr_h, lr_w = 32, 96

    train_hr = tmp_path / "train" / "hr"
    train_lr = tmp_path / "train" / "lr_x4"
    train_geom = tmp_path / "train" / "geometry"
    test_hr = tmp_path / "test" / "flickr1024" / "hr"
    test_lr = tmp_path / "test" / "flickr1024" / "lr_x4"

    for d in (train_hr, train_lr, train_geom, test_hr, test_lr):
        d.mkdir(parents=True)

    for stem in ("0001", "0002"):
        for suffix in ("_L.png", "_R.png"):
            _save_rgb(train_hr / f"{stem}{suffix}", hr_h, hr_w)
            _save_rgb(train_lr / f"{stem}{suffix}", lr_h, lr_w)

    for stem in ("0001", "0002"):
        np.savez(
            train_geom / f"{stem}.npz",
            disp_left=np.random.rand(lr_h, lr_w).astype(np.float32),
            disp_right=np.random.rand(lr_h, lr_w).astype(np.float32),
            conf_left=np.random.rand(lr_h, lr_w).astype(np.float32),
            conf_right=np.random.rand(lr_h, lr_w).astype(np.float32),
        )

    for suffix in ("_L.png", "_R.png"):
        _save_rgb(test_hr / f"0001{suffix}", hr_h, hr_w)
        _save_rgb(test_lr / f"0001{suffix}", lr_h, lr_w)

    return tmp_path
