from __future__ import annotations

from math import ceil
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

    F64 = npt.NDArray[np.float64]
    I32 = npt.NDArray[np.int32]
    U8 = npt.NDArray[np.uint8]


def _cubic(x: F64) -> F64:
    ax = np.absolute(x)
    ax2 = ax * ax
    ax3 = ax2 * ax
    return (  # type: ignore[no-any-return]
        (1.5 * ax3 - 2.5 * ax2 + 1) * (ax <= 1)
        + (-0.5 * ax3 + 2.5 * ax2 - 4 * ax + 2) * ((ax > 1) & (ax <= 2))
    )


def _contributions(in_len: int, out_len: int, scale: float) -> tuple[F64, I32]:
    kw = 4.0 / scale if scale < 1 else 4.0
    kernel = (lambda x: scale * _cubic(scale * x)) if scale < 1 else _cubic

    x = np.arange(1, out_len + 1, dtype=np.float64)
    u = x / scale + 0.5 * (1 - 1 / scale)
    left = np.floor(u - kw / 2)
    ind = (np.expand_dims(left, 1) + np.arange(int(ceil(kw)) + 2) - 1).astype(np.int32)

    weights = kernel(np.expand_dims(u, 1) - ind - 1)
    weights /= np.sum(weights, axis=1, keepdims=True)

    mirror = np.concatenate((np.arange(in_len), np.arange(in_len - 1, -1, step=-1))).astype(
        np.int32
    )
    ind = mirror[np.mod(ind, mirror.size)]

    keep = np.nonzero(np.any(weights, axis=0))
    return weights[:, keep], ind[:, keep]


def _apply(img: F64, dim: int, weights: F64, indices: I32) -> F64:
    s = weights.shape
    if dim == 0:
        w = weights.reshape(s[0], s[2], 1, 1)
        return np.sum(w * img[indices].squeeze(1).astype(np.float64), axis=1)  # type: ignore[no-any-return]
    w = weights.reshape(1, s[0], s[2], 1)
    return np.sum(w * img[:, indices].squeeze(2).astype(np.float64), axis=2)  # type: ignore[no-any-return]


def matlab_imresize(img: U8, scale: float) -> U8:
    out_hw = [int(ceil(scale * img.shape[k])) for k in range(2)]
    scales = [out_hw[k] / img.shape[k] for k in range(2)]
    order = np.argsort(scales)

    contrib = [_contributions(img.shape[k], out_hw[k], scales[k]) for k in range(2)]

    buf = img.astype(np.float64)
    if buf.ndim == 2:
        buf = buf[:, :, np.newaxis]
        was_2d = True
    else:
        was_2d = False

    for k in range(2):
        d = int(order[k])
        buf = _apply(buf, d, contrib[d][0], contrib[d][1])

    if was_2d:
        buf = buf[:, :, 0]

    return np.clip(np.around(buf), 0, 255).astype(np.uint8)
