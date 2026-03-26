from __future__ import annotations

from collections.abc import Sequence

import pytest
import torch
from torch import Tensor

from sissr.baselines.diffssr.model import DIFFSSR, MultiheadDiffAttn


def _make_model(window_size: int = 16) -> DIFFSSR:
    return DIFFSSR(
        embed_dim=180,
        depths=(3,) * 13,
        num_heads=(6,) * 13,
        window_size=window_size,
        mlp_ratio=2.0,
        upscale=4,
        img_range=1.0,
        resi_connection="1conv",
    )


def _make_attn(window_size: int = 4, embed_dim: int = 48) -> MultiheadDiffAttn:
    return MultiheadDiffAttn(
        embed_dim=embed_dim,
        window_size=(window_size, window_size),
        num_heads=6,
    )


def _make_rpi(window_size: int) -> Tensor:
    coords_h = torch.arange(window_size)
    coords_w = torch.arange(window_size)
    coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
    coords_flatten = torch.flatten(coords, 1)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += window_size - 1
    relative_coords[:, :, 1] += window_size - 1
    relative_coords[:, :, 0] *= 2 * window_size - 1
    result: Tensor = relative_coords.sum(-1)
    return result


def _make_shift_mask(window_size: int, x_size: tuple[int, int]) -> Tensor:
    h, w = x_size
    shift_size = window_size // 2
    img_mask = torch.zeros((1, h, w, 1))
    h_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )
    w_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )
    cnt = 0
    for hs in h_slices:
        for ws in w_slices:
            img_mask[:, hs, ws, :] = cnt
            cnt += 1

    mask_windows = img_mask.view(
        1, h // window_size, window_size, w // window_size, window_size, 1
    )
    mask_windows = mask_windows.permute(0, 1, 3, 2, 4, 5).contiguous()
    mask_windows = mask_windows.view(-1, window_size * window_size)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    result = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
    return result


def test_forward_pass_shape() -> None:
    model = _make_model()
    model.eval()
    x = torch.randn(1, 6, 64, 64)
    with torch.no_grad():
        out: Tensor = model(x)
    assert out.shape == (1, 6, 256, 256)
    assert out.isfinite().all()
    assert not torch.allclose(out, torch.zeros_like(out))


def test_parameter_count() -> None:
    model = _make_model()
    total = sum(p.numel() for p in model.parameters())
    assert abs(total - 20_000_000) < 100_000, f"param count {total} not ~20M"


def test_non_divisible_input_pads_and_crops() -> None:
    model = _make_model(window_size=16)
    model.eval()
    x = torch.randn(1, 6, 30, 90)
    with torch.no_grad():
        out: Tensor = model(x)
    assert out.shape == (1, 6, 120, 360)
    assert out.isfinite().all()


def test_nonsquare_input() -> None:
    model = _make_model()
    model.eval()
    x = torch.randn(1, 6, 48, 96)
    with torch.no_grad():
        out: Tensor = model(x)
    assert out.shape == (1, 6, 192, 384)
    assert out.isfinite().all()


def test_small_input_at_window_boundary() -> None:
    model = _make_model(window_size=16)
    model.eval()
    x = torch.randn(1, 6, 16, 16)
    with torch.no_grad():
        out: Tensor = model(x)
    assert out.shape == (1, 6, 64, 64)
    assert out.isfinite().all()
    assert not torch.allclose(out, torch.zeros_like(out))


def test_small_input_disables_nonzero_spatial_roll(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _make_model(window_size=16)
    model.eval()
    spatial_rolls: list[tuple[int, ...]] = []
    original_roll = torch.roll

    def recording_roll(
        input_tensor: Tensor,
        shifts: int | Sequence[int],
        dims: int | Sequence[int] | None = None,
    ) -> Tensor:
        shift_tuple = (shifts,) if isinstance(shifts, int) else tuple(shifts)
        dim_tuple = None if dims is None else (dims,) if isinstance(dims, int) else tuple(dims)
        if dim_tuple == (1, 2) and any(shift != 0 for shift in shift_tuple):
            spatial_rolls.append(shift_tuple)
        return original_roll(input_tensor, shifts, dims)

    monkeypatch.setattr(torch, "roll", recording_roll)
    x = torch.randn(1, 6, 16, 16)
    with torch.no_grad():
        _ = model(x)
    assert spatial_rolls == []


def test_batch_size_greater_than_one() -> None:
    model = _make_model()
    model.eval()
    x = torch.randn(2, 6, 32, 32)
    with torch.no_grad():
        out: Tensor = model(x)
    assert out.shape == (2, 6, 128, 128)
    assert out.isfinite().all()


def test_output_dtype_matches_input() -> None:
    model = _make_model()
    model.eval()
    x = torch.randn(1, 6, 64, 64)
    with torch.no_grad():
        out: Tensor = model(x)
    assert out.dtype == x.dtype


def test_left_right_not_identical() -> None:
    model = _make_model()
    model.eval()
    left = torch.randn(1, 3, 32, 32)
    right = torch.randn(1, 3, 32, 32)
    x = torch.cat([left, right], dim=1)
    with torch.no_grad():
        out: Tensor = model(x)
    out_l = out[:, :3]
    out_r = out[:, 3:]
    assert not torch.allclose(out_l, out_r, atol=1e-4)


def test_identity_resi_connection() -> None:
    model = DIFFSSR(
        embed_dim=180,
        depths=(3,) * 13,
        num_heads=(6,) * 13,
        window_size=16,
        mlp_ratio=2.0,
        upscale=4,
        img_range=1.0,
        resi_connection="identity",
    )
    model.eval()
    x = torch.randn(1, 6, 32, 32)
    with torch.no_grad():
        out: Tensor = model(x)
    assert out.shape == (1, 6, 128, 128)


def test_unsupported_upscale_raises() -> None:
    with pytest.raises(ValueError, match="scale 5 is not supported"):
        DIFFSSR(
            embed_dim=180,
            depths=(3,) * 13,
            num_heads=(6,) * 13,
            window_size=16,
            upscale=5,
        )


def test_all_parameters_receive_gradients() -> None:
    model = _make_model()
    model.train()
    x = torch.randn(1, 6, 32, 32, requires_grad=False)
    out: Tensor = model(x)
    loss = out.sum()
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"no gradient for {name}"


def test_multihead_diff_attn_masked_path_shape_and_finite() -> None:
    torch.manual_seed(1)
    attn = _make_attn()
    x = torch.randn(8, 16, 48)
    rpi = _make_rpi(4)
    mask = _make_shift_mask(4, (8, 8))

    out = attn(x, rpi, mask)

    assert out.shape == x.shape
    assert out.isfinite().all()
    assert not torch.allclose(out, torch.zeros_like(out))


def test_multihead_diff_attn_mask_batch_mismatch_raises() -> None:
    attn = _make_attn()
    x = torch.randn(3, 16, 48)
    rpi = _make_rpi(4)
    mask = _make_shift_mask(4, (8, 8))

    with pytest.raises(ValueError, match="must be divisible by attention mask windows"):
        attn(x, rpi, mask)


def test_multihead_diff_attn_odd_heads_raise() -> None:
    with pytest.raises(ValueError, match="must be even"):
        MultiheadDiffAttn(embed_dim=48, window_size=(4, 4), num_heads=5)


def test_multihead_diff_attn_autocast_masked_path_is_finite() -> None:
    attn = _make_attn().train()
    x = torch.randn(8, 16, 48, requires_grad=True)
    rpi = _make_rpi(4)
    mask = _make_shift_mask(4, (8, 8))

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = attn(x, rpi, mask)
        loss = out.square().mean()

    loss.backward()

    assert out.isfinite().all()
    assert torch.isfinite(loss)
    assert x.grad is not None
    assert x.grad.isfinite().all()
