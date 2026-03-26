from __future__ import annotations

import numpy as np

from sissr.data.matlab_resize import matlab_imresize


def test_output_shape_downscale() -> None:
    img = np.random.randint(0, 256, (64, 128, 3), dtype=np.uint8)
    out = matlab_imresize(img, 0.5)
    assert out.shape == (32, 64, 3)


def test_output_shape_upscale() -> None:
    img = np.random.randint(0, 256, (32, 64, 3), dtype=np.uint8)
    out = matlab_imresize(img, 2.0)
    assert out.shape == (64, 128, 3)


def test_output_dtype_is_uint8() -> None:
    img = np.random.randint(0, 256, (16, 32, 3), dtype=np.uint8)
    assert matlab_imresize(img, 0.5).dtype == np.uint8
    assert matlab_imresize(img, 2.0).dtype == np.uint8


def test_output_range_valid() -> None:
    img = np.random.randint(0, 256, (64, 128, 3), dtype=np.uint8)
    out = matlab_imresize(img, 0.25)
    assert out.min() >= 0
    assert out.max() <= 255


def test_grayscale_input() -> None:
    img = np.random.randint(0, 256, (64, 128), dtype=np.uint8)
    out = matlab_imresize(img, 0.5)
    assert out.shape == (32, 64)
    assert out.ndim == 2


def test_scale_one_is_identity() -> None:
    img = np.random.randint(0, 256, (32, 64, 3), dtype=np.uint8)
    out = matlab_imresize(img, 1.0)
    assert out.shape == img.shape
    np.testing.assert_array_equal(out, img)


def test_uniform_image_stays_uniform() -> None:
    img = np.full((64, 128, 3), 127, dtype=np.uint8)
    out = matlab_imresize(img, 0.5)
    np.testing.assert_array_equal(out, np.full((32, 64, 3), 127, dtype=np.uint8))


def test_quarter_scale_known_output() -> None:
    img = np.zeros((8, 8, 1), dtype=np.uint8)
    img[:4, :, :] = 255
    out = matlab_imresize(img, 0.5)
    assert out.shape == (4, 4, 1)
    assert out[0, 0, 0] > 200
    assert out[3, 0, 0] < 55
