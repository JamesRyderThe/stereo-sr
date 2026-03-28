from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor

from sissr.models.enums import StereoDirection
from sissr.models.layers.attn import Attention
from sissr.models.layers.depth_agg import DepthAggregator, DepthState
from sissr.models.layers.embedding import (
    AxisConfig,
    EpipolarRoPE,
    RectifiedDisparityRoPE,
    RotarySpec,
    SpatialRoPE,
    VisionGrid,
)
from sissr.models.layers.norm import ChannelRMSNorm
from sissr.models.layers.window_attn import (
    WindowedAttention,
    WindowedCrossAttention,
    build_shift_mask,
)
from sissr.models.stereo_sr import StereoDepthPair


def _baseline_shift_mask(
    window_size: tuple[int, int],
    shift_size: tuple[int, int],
    x_size: tuple[int, int],
) -> Tensor:
    h, w = x_size
    window_h, window_w = window_size
    shift_h, shift_w = shift_size
    img_mask = torch.zeros((1, h, w, 1))
    h_slices = (
        slice(0, -window_h),
        slice(-window_h, -shift_h),
        slice(-shift_h, None),
    )
    w_slices = (
        slice(0, -window_w),
        slice(-window_w, -shift_w),
        slice(-shift_w, None),
    )
    count = 0
    for hs in h_slices:
        for ws in w_slices:
            img_mask[:, hs, ws, :] = count
            count += 1

    mask_windows = img_mask.view(
        1,
        h // window_h,
        window_h,
        w // window_w,
        window_w,
        1,
    )
    mask_windows = mask_windows.permute(0, 1, 3, 2, 4, 5).contiguous()
    mask_windows = mask_windows.view(-1, window_h * window_w)
    return mask_windows.unsqueeze(1) != mask_windows.unsqueeze(2)


@pytest.mark.parametrize(
    ("window_size", "x_size"),
    [
        ((4, 4), (8, 8)),
        ((4, 4), (8, 12)),
        ((4, 4), (12, 12)),
        ((4, 8), (8, 16)),
    ],
)
def test_build_shift_mask_matches_baseline(
    window_size: tuple[int, int], x_size: tuple[int, int]
) -> None:
    shift_size = (window_size[0] // 2, window_size[1] // 2)
    expected = _baseline_shift_mask(window_size, shift_size, x_size)
    actual = build_shift_mask(
        x_size[0],
        x_size[1],
        window_size,
        shift_size,
        device=torch.device("cpu"),
    )
    assert torch.equal(actual, expected)


def test_build_shift_mask_invalid_spatial_raises() -> None:
    with pytest.raises(ValueError, match="not divisible"):
        build_shift_mask(10, 8, (4, 8), (2, 4), device=torch.device("cpu"))


def test_channel_rms_norm_shape_dtype_and_finite() -> None:
    norm = ChannelRMSNorm(6)
    x = torch.randn(2, 6, 4, 5)

    out = norm(x)

    assert out.shape == x.shape
    assert out.dtype == x.dtype
    assert out.isfinite().all()


def test_attention_spatial_rope_shape_and_finite() -> None:
    rope = SpatialRoPE(AxisConfig.from_head_dim(8))
    spec = RotarySpec(rope=rope, grid=VisionGrid(height=8, width=8))
    module = Attention(embed_dim=32, num_heads=4)
    x = torch.randn(2, 64, 32)

    out = module(x, rotary=spec)

    assert out.shape == x.shape
    assert out.isfinite().all()


def test_windowed_attention_shifted_spatial_rope_shape_and_finite() -> None:
    rope = SpatialRoPE(AxisConfig.from_head_dim(8))
    spec = RotarySpec(rope=rope, grid=VisionGrid(height=8, width=8))
    module = WindowedAttention(embed_dim=32, num_heads=4, window_size=4)
    x = torch.randn(1, 32, 8, 8)

    out = module(x, shift=True, rotary=spec)

    assert isinstance(out, Tensor)
    assert out.shape == x.shape
    assert out.isfinite().all()


@pytest.mark.parametrize("shift", [False, True])
def test_windowed_cross_attention_shape_and_finite(shift: bool) -> None:
    module = WindowedCrossAttention(embed_dim=32, num_heads=4, window_size=(4, 8))
    left = torch.randn(1, 32, 8, 16)
    right = torch.randn(1, 32, 8, 16)

    left_out, right_out = module(left, right, shift=shift)

    assert left_out.shape == left.shape
    assert right_out.shape == right.shape
    assert left_out.isfinite().all()
    assert right_out.isfinite().all()


def test_windowed_cross_attention_tied_weights_swap_outputs() -> None:
    torch.manual_seed(0)
    module = WindowedCrossAttention(embed_dim=32, num_heads=4, window_size=(4, 8))
    module.eval()
    with torch.no_grad():
        module.right_match_proj.weight.copy_(module.left_match_proj.weight)
        assert module.left_match_proj.bias is not None
        assert module.right_match_proj.bias is not None
        module.right_match_proj.bias.copy_(module.left_match_proj.bias)
        module.right_value_proj.weight.copy_(module.left_value_proj.weight)
        assert module.left_value_proj.bias is not None
        assert module.right_value_proj.bias is not None
        module.right_value_proj.bias.copy_(module.left_value_proj.bias)
        module.right_out_proj.weight.copy_(module.left_out_proj.weight)
        assert module.left_out_proj.bias is not None
        assert module.right_out_proj.bias is not None
        module.right_out_proj.bias.copy_(module.left_out_proj.bias)
        module.gamma.copy_(module.beta)
    left = torch.randn(1, 32, 8, 16)
    right = torch.randn(1, 32, 8, 16)

    left_out, right_out = module(left, right)
    swapped_left_out, swapped_right_out = module(right, left)

    assert torch.allclose(left_out, swapped_right_out, atol=1e-5)
    assert torch.allclose(right_out, swapped_left_out, atol=1e-5)


def test_windowed_cross_attention_default_projections_are_untied() -> None:
    module = WindowedCrossAttention(embed_dim=32, num_heads=4, window_size=(4, 8))

    assert module.left_match_proj.weight.data_ptr() != module.right_match_proj.weight.data_ptr()
    assert module.left_value_proj.weight.data_ptr() != module.right_value_proj.weight.data_ptr()
    assert module.left_out_proj.weight.data_ptr() != module.right_out_proj.weight.data_ptr()


def test_windowed_cross_attention_invalid_rectangular_spatial_raises() -> None:
    module = WindowedCrossAttention(embed_dim=32, num_heads=4, window_size=(4, 8))
    x = torch.randn(1, 32, 8, 12)

    with pytest.raises(ValueError, match="must divide by window_size"):
        module(x, x)


def test_windowed_cross_attention_mismatched_shapes_raise() -> None:
    module = WindowedCrossAttention(embed_dim=32, num_heads=4, window_size=(4, 8))
    left = torch.randn(1, 32, 8, 16)
    right = torch.randn(1, 32, 8, 8)

    with pytest.raises(ValueError, match="must have the same shape"):
        module(left, right)


def test_windowed_cross_attention_beta_can_disable_left_transfer() -> None:
    module = WindowedCrossAttention(embed_dim=32, num_heads=4, window_size=(4, 8))
    left = torch.randn(1, 32, 8, 16)
    right = torch.randn(1, 32, 8, 16)

    with torch.no_grad():
        module.beta.zero_()

    left_out, right_out = module(left, right)

    assert torch.equal(left_out, torch.zeros_like(left_out))
    assert not torch.equal(right_out, torch.zeros_like(right_out))


def test_windowed_cross_attention_gamma_can_disable_right_transfer() -> None:
    module = WindowedCrossAttention(embed_dim=32, num_heads=4, window_size=(4, 8))
    left = torch.randn(1, 32, 8, 16)
    right = torch.randn(1, 32, 8, 16)

    with torch.no_grad():
        module.gamma.zero_()

    left_out, right_out = module(left, right)

    assert not torch.equal(left_out, torch.zeros_like(left_out))
    assert torch.equal(right_out, torch.zeros_like(right_out))


def test_windowed_cross_attention_sharp_matches_get_higher_streamed_confidence() -> None:
    module = WindowedCrossAttention(embed_dim=4, num_heads=1, window_size=(1, 2))
    sharp = torch.tensor([[[[4.0, 0.0], [0.0, 4.0]]]])
    diffuse = torch.ones((1, 1, 2, 2))

    sharp_left_lse, sharp_left_max = module._row_stats(sharp, sharp, None)
    sharp_right_lse, sharp_right_max = module._row_stats(sharp, sharp, None)
    diffuse_left_lse, diffuse_left_max = module._row_stats(diffuse, diffuse, None)
    diffuse_right_lse, diffuse_right_max = module._row_stats(diffuse, diffuse, None)

    sharp_left_conf, sharp_right_conf = module._stream_cycle_confidence(
        left_q=sharp,
        right_q=sharp,
        left_k=sharp,
        right_k=sharp,
        left_attn_mask=None,
        right_attn_mask=None,
        left_row_lse=sharp_left_lse,
        left_row_max=sharp_left_max,
        right_row_lse=sharp_right_lse,
        right_row_max=sharp_right_max,
    )
    diffuse_left_conf, diffuse_right_conf = module._stream_cycle_confidence(
        left_q=diffuse,
        right_q=diffuse,
        left_k=diffuse,
        right_k=diffuse,
        left_attn_mask=None,
        right_attn_mask=None,
        left_row_lse=diffuse_left_lse,
        left_row_max=diffuse_left_max,
        right_row_lse=diffuse_right_lse,
        right_row_max=diffuse_right_max,
    )

    assert torch.all(sharp_left_conf >= 0.0)
    assert torch.all(sharp_left_conf <= 1.0)
    assert torch.all(sharp_right_conf >= 0.0)
    assert torch.all(sharp_right_conf <= 1.0)
    assert torch.all(sharp_left_conf > diffuse_left_conf)
    assert torch.all(sharp_right_conf > diffuse_right_conf)


def test_windowed_cross_attention_per_head_mask_changes_output() -> None:
    torch.manual_seed(0)
    module = WindowedCrossAttention(embed_dim=32, num_heads=4, window_size=(4, 8))
    left = torch.randn(1, 32, 8, 16)
    right = torch.randn(1, 32, 8, 16)
    seq_len = 32
    num_direction_windows = (left.shape[2] // 4) * (left.shape[3] // 8)
    base_mask = torch.zeros((num_direction_windows, 4, seq_len, seq_len))
    masked = base_mask.clone()
    masked[:, 1, :, seq_len // 2 :] = torch.finfo(masked.dtype).min

    base_left, base_right = module(left, right, mask=base_mask)
    masked_left, masked_right = module(left, right, mask=masked)

    assert torch.max(torch.abs(base_left - masked_left)).item() > 0.0
    assert torch.max(torch.abs(base_right - masked_right)).item() > 0.0


def test_windowed_cross_attention_tau_changes_output() -> None:
    torch.manual_seed(0)
    module = WindowedCrossAttention(embed_dim=32, num_heads=4, window_size=(4, 8))
    module.eval()
    left = torch.randn(1, 32, 8, 16)
    right = torch.randn(1, 32, 8, 16)

    with torch.no_grad():
        module.log_tau_h.fill_(math.log(0.25))
        low_temp_left, _ = module(left, right)
        module.log_tau_h.fill_(math.log(4.0))
        high_temp_left, _ = module(left, right)

    assert torch.max(torch.abs(low_temp_left - high_temp_left)).item() > 0.0


def test_windowed_cross_attention_extreme_tau_stays_finite() -> None:
    module = WindowedCrossAttention(embed_dim=32, num_heads=4, window_size=(4, 8))
    left = torch.randn(1, 32, 8, 16)
    right = torch.randn(1, 32, 8, 16)

    with torch.no_grad():
        module.log_tau_h.fill_(100.0)
        high_out = module(left, right)
        module.log_tau_h.fill_(-100.0)
        low_out = module(left, right)

    assert high_out[0].isfinite().all()
    assert high_out[1].isfinite().all()
    assert low_out[0].isfinite().all()
    assert low_out[1].isfinite().all()


def test_windowed_cross_attention_bfloat16_autocast_stays_finite() -> None:
    module = WindowedCrossAttention(embed_dim=32, num_heads=4, window_size=(4, 8))
    left = torch.randn(1, 32, 8, 16)
    right = torch.randn(1, 32, 8, 16)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        left_out, right_out = module(left, right)

    assert left_out.isfinite().all()
    assert right_out.isfinite().all()


def test_depth_state_accumulate_sets_partial_when_none() -> None:
    agg = DepthAggregator(dim=8, num_sublayers=1, max_sources=2)
    state = DepthState.create(agg, torch.ones(2, 4, 8))
    state.commit_boundary()

    assert state.partial is None
    state.accumulate(torch.full((2, 4, 8), 3.0))
    assert state.partial is not None
    assert torch.equal(state.partial, torch.full((2, 4, 8), 3.0))


def test_depth_state_accumulate_adds_to_existing_partial() -> None:
    agg = DepthAggregator(dim=8, num_sublayers=1, max_sources=2)
    state = DepthState.create(agg, torch.ones(2, 4, 8))
    state.commit_boundary()

    state.accumulate(torch.full((2, 4, 8), 2.0))
    state.accumulate(torch.full((2, 4, 8), 3.0))
    assert state.partial is not None
    assert torch.equal(state.partial, torch.full((2, 4, 8), 5.0))


def test_depth_state_finalize_includes_initial_source_by_default() -> None:
    agg = DepthAggregator(dim=8, num_sublayers=1, max_sources=2)
    state = DepthState.create(agg, torch.ones(2, 4, 8))
    state.commit_boundary()

    state.accumulate(torch.full((2, 4, 8), 2.0))
    state.accumulate(torch.full((2, 4, 8), 3.0))
    result = state.finalize()

    assert torch.equal(result, torch.full((2, 4, 8), 3.0))


def test_depth_state_finalize_skip_first_excludes_initial_source() -> None:
    agg = DepthAggregator(dim=8, num_sublayers=1, max_sources=2)
    state = DepthState.create(agg, torch.ones(2, 4, 8))
    state.commit_boundary()

    state.accumulate(torch.full((2, 4, 8), 2.0))
    state.accumulate(torch.full((2, 4, 8), 3.0))
    result = state.finalize(skip_first=True)

    assert torch.equal(result, torch.full((2, 4, 8), 5.0))


def test_depth_state_finalize_averages_logical_blocks_with_zero_init() -> None:
    agg = DepthAggregator(dim=8, num_sublayers=1, max_sources=3)
    state = DepthState.create(agg, torch.ones(2, 4, 8))
    state.commit_boundary()

    state.accumulate(torch.full((2, 4, 8), 2.0))
    state.commit_boundary()
    state.accumulate(torch.full((2, 4, 8), 4.0))
    result = state.finalize(skip_first=True)

    assert torch.equal(result, torch.full((2, 4, 8), 3.0))


def test_depth_state_commit_boundary_raises_when_no_partial() -> None:
    agg = DepthAggregator(dim=8, num_sublayers=1, max_sources=2)
    state = DepthState.create(agg, torch.ones(2, 4, 8))
    state.commit_boundary()

    with pytest.raises(RuntimeError, match="commit_boundary called with no partial"):
        state.commit_boundary()


def test_depth_state_finalize_raises_when_no_partial() -> None:
    agg = DepthAggregator(dim=8, num_sublayers=1, max_sources=2)
    state = DepthState.create(agg, torch.ones(2, 4, 8))
    state.commit_boundary()

    with pytest.raises(RuntimeError, match="finalize called with no partial"):
        state.finalize()


def test_depth_aggregator_weighted_sum_with_zero_init_is_uniform() -> None:
    agg = DepthAggregator(dim=4, num_sublayers=1, max_sources=2)
    source_a = torch.full((1, 2, 4), 10.0)
    source_b = torch.full((1, 2, 4), 20.0)
    sources = torch.stack([source_a, source_b])

    result = agg(sources, num_sources=2, sublayer_idx=0)

    assert result.shape == (1, 2, 4)
    assert torch.allclose(result, torch.full((1, 2, 4), 15.0), atol=1e-5)


def test_depth_aggregator_padded_matches_unpadded() -> None:
    source_a = torch.tensor([[[1.0, 2.0, 0.0, -1.0], [0.0, 1.0, 1.0, 0.0]]])
    source_b = torch.tensor([[[2.0, -1.0, 3.0, 0.5], [1.0, 0.0, -2.0, 2.0]]])
    tight = DepthAggregator(dim=4, num_sublayers=1, max_sources=2)
    wide = DepthAggregator(dim=4, num_sublayers=1, max_sources=4)
    wide.load_state_dict(tight.state_dict())
    with torch.no_grad():
        query = torch.tensor([1.0, -0.5, 0.75, -0.25])
        tight.queries[0].copy_(query)
        wide.queries[0].copy_(query)
    tight_sources = torch.stack([source_a, source_b])
    wide_sources = torch.cat([tight_sources, torch.zeros(2, 1, 2, 4)], dim=0)

    tight_out = tight(tight_sources, num_sources=2, sublayer_idx=0)
    wide_out = wide(wide_sources, num_sources=2, sublayer_idx=0)

    assert torch.allclose(tight_out, wide_out, atol=1e-5)


def test_depth_aggregator_mask_gives_zero_weight() -> None:
    agg = DepthAggregator(dim=4, num_sublayers=1, max_sources=4)
    source = torch.full((1, 2, 4), 10.0)
    sources = torch.cat([source.unsqueeze(0), torch.zeros(3, 1, 2, 4)], dim=0)

    result = agg(sources, num_sources=1, sublayer_idx=0)

    assert torch.equal(result, source)


def test_depth_aggregator_single_source_padded_finite() -> None:
    agg = DepthAggregator(dim=4, num_sublayers=1, max_sources=8)
    source = torch.randn(1, 2, 4)
    sources = torch.cat([source.unsqueeze(0), torch.zeros(7, 1, 2, 4)], dim=0)

    result = agg(sources, num_sources=1, sublayer_idx=0)

    assert result.isfinite().all()


def test_depth_state_sublayer_idx_advances_per_aggregate() -> None:
    num_sublayers = 6
    agg = DepthAggregator(dim=4, num_sublayers=num_sublayers, max_sources=2)
    state = DepthState.create(agg, torch.randn(1, 2, 4))
    state.commit_boundary()

    for _ in range(num_sublayers):
        state.aggregate()
        state.accumulate(torch.randn(1, 2, 4))

    assert state._sublayer_idx == num_sublayers


def test_depth_state_finalize_consumes_extra_query() -> None:
    agg = DepthAggregator(dim=4, num_sublayers=2, max_sources=2)
    state = DepthState.create(agg, torch.ones(1, 2, 4))
    state.commit_boundary()

    state.aggregate()
    state.accumulate(torch.full((1, 2, 4), 2.0))
    state.finalize()

    assert state._sublayer_idx == 2


def test_depth_state_finalize_uses_finalize_query_row() -> None:
    agg = DepthAggregator(dim=2, num_sublayers=2, max_sources=3)
    with torch.no_grad():
        agg.queries[0].copy_(torch.tensor([1.0, 0.0]))
        agg.queries[1].copy_(torch.tensor([0.0, 1.0]))
    initial = torch.tensor([[[6.0, 0.0]]])
    block_a = torch.tensor([[[3.0, 0.0]]])
    block_b = torch.tensor([[[0.0, 5.0]]])
    state = DepthState.create(agg, initial)
    state.commit_boundary()

    state.aggregate()
    state.accumulate(block_a)
    state.commit_boundary()
    state.accumulate(block_b)
    result = state.finalize(skip_first=True)

    final_sources = torch.stack([block_a, block_b])
    normed = agg.source_norm(final_sources)
    logits = torch.einsum("nbsd, d -> nbs", normed, agg.queries[1])
    expected = torch.einsum("nbsd, nbs -> bsd", final_sources, logits.softmax(dim=0))

    assert torch.allclose(result, expected, atol=1e-5)
    assert not torch.allclose(result, (block_a + block_b) / 2, atol=1e-5)


def test_stereo_depth_pair_commit_and_finalize_both_views() -> None:
    left_agg = DepthAggregator(dim=4, num_sublayers=1, max_sources=2)
    right_agg = DepthAggregator(dim=4, num_sublayers=1, max_sources=2)
    pair = StereoDepthPair.create(
        left_agg,
        right_agg,
        torch.full((1, 2, 4), 1.0),
        torch.full((1, 2, 4), 10.0),
    )
    pair.commit_boundary()

    pair.left.accumulate(torch.full((1, 2, 4), 2.0))
    pair.right.accumulate(torch.full((1, 2, 4), 20.0))

    left_out, right_out = pair.finalize(skip_first=True)

    assert torch.equal(left_out, torch.full((1, 2, 4), 2.0))
    assert torch.equal(right_out, torch.full((1, 2, 4), 20.0))


def test_epipolar_rope_q_shape_and_finite() -> None:
    rope = EpipolarRoPE(head_dim=8, embed_dim=32)
    q = torch.randn(2, 32, 4, 8)

    out = rope.apply_q(q, height=4, width=8)

    assert out.shape == q.shape
    assert out.isfinite().all()


def test_epipolar_rope_k_shift_changes_output() -> None:
    rope = EpipolarRoPE(head_dim=8, embed_dim=32)
    k = torch.randn(2, 32, 4, 8)
    mu_zero = torch.zeros(2, 32)
    mu_nonzero = torch.full((2, 32), 5.0)
    sigma = torch.ones(2, 32)

    out_zero = rope.apply_k(k, height=4, width=8, mu=mu_zero, sigma=sigma)
    out_shifted = rope.apply_k(k, height=4, width=8, mu=mu_nonzero, sigma=sigma)

    assert not torch.allclose(out_zero, out_shifted)


def test_epipolar_rope_shift_only_affects_width_dims() -> None:
    rope = EpipolarRoPE(head_dim=8, embed_dim=32)
    k = torch.randn(2, 32, 4, 8)
    mu_zero = torch.zeros(2, 32)
    mu_nonzero = torch.full((2, 32), 5.0)
    sigma = torch.ones(2, 32)

    out_zero = rope.apply_k(k, height=4, width=8, mu=mu_zero, sigma=sigma)
    out_shifted = rope.apply_k(k, height=4, width=8, mu=mu_nonzero, sigma=sigma)

    assert torch.allclose(out_zero[..., : rope.h_dim], out_shifted[..., : rope.h_dim], atol=1e-5)
    assert not torch.allclose(out_zero[..., rope.h_dim :], out_shifted[..., rope.h_dim :])


def test_epipolar_rope_sinc_dampening_reduces_high_freq_magnitude() -> None:
    rope = EpipolarRoPE(head_dim=8, embed_dim=32)
    k = torch.randn(2, 32, 4, 8)
    mu = torch.zeros(2, 32)
    sigma_small = torch.full((2, 32), 0.01)
    sigma_large = torch.full((2, 32), 100.0)

    out_small = rope.apply_k(k, height=4, width=8, mu=mu, sigma=sigma_small)
    out_large = rope.apply_k(k, height=4, width=8, mu=mu, sigma=sigma_large)

    w_small = out_small[..., rope.h_dim :]
    w_large = out_large[..., rope.h_dim :]
    assert w_large.abs().mean() < w_small.abs().mean()

    h_small = out_small[..., : rope.h_dim]
    h_large = out_large[..., : rope.h_dim]
    assert torch.allclose(h_small, h_large, atol=1e-5)


def test_epipolar_rope_zero_sigma_matches_q_at_same_positions() -> None:
    rope = EpipolarRoPE(head_dim=8, embed_dim=32)
    tensor = torch.randn(2, 32, 4, 8)
    mu = torch.zeros(2, 32)
    sigma = torch.zeros(2, 32)

    q_out = rope.apply_q(tensor, height=4, width=8)
    k_out = rope.apply_k(tensor, height=4, width=8, mu=mu, sigma=sigma)

    assert torch.allclose(q_out, k_out, atol=1e-5)


def test_epipolar_rope_invalid_head_dim_raises() -> None:
    with pytest.raises(ValueError, match="head_dim must be even"):
        EpipolarRoPE(head_dim=7, embed_dim=32)


def test_epipolar_rope_invalid_init_sigma_raises() -> None:
    with pytest.raises(ValueError, match="init_sigma must be positive"):
        EpipolarRoPE(head_dim=8, embed_dim=32, init_sigma=0.0)
    with pytest.raises(ValueError, match="init_sigma must be positive"):
        EpipolarRoPE(head_dim=8, embed_dim=32, init_sigma=-1.0)


def test_epipolar_rope_too_many_height_pairs_raises() -> None:
    with pytest.raises(ValueError, match="height_pairs.*must be less than"):
        EpipolarRoPE(head_dim=8, embed_dim=32, height_pairs=4)


def test_epipolar_rope_init_sigma_matches_requested_value() -> None:
    rope = EpipolarRoPE(head_dim=8, embed_dim=32, init_sigma=2.0)
    dummy = torch.zeros(1, 1, 32)
    _, sigma = rope.predict_offset(dummy)
    assert abs(sigma.item() - 2.0) < 1e-4


def test_epipolar_rope_mu_bounded_to_max_shift() -> None:
    rope = EpipolarRoPE(head_dim=8, embed_dim=32, max_shift=10.0)
    tokens = torch.randn(2, 16, 32) * 100.0
    mu, _ = rope.predict_offset(tokens)
    assert mu.abs().max().item() <= 10.0 + 1e-6


def test_epipolar_rope_grid_cache_respects_width() -> None:
    rope = EpipolarRoPE(head_dim=8, embed_dim=32)
    q = torch.randn(2, 32, 4, 8)

    rope.apply_q(q, height=4, width=8)
    cols_8 = rope._cached_cols.clone()

    q_alt = torch.randn(2, 32, 4, 8)
    rope.apply_q(q_alt, height=8, width=4)
    cols_4 = rope._cached_cols.clone()

    assert cols_8.shape == cols_4.shape
    assert not torch.equal(cols_8, cols_4)


def test_epipolar_rope_direction_flips_shift() -> None:
    rope = EpipolarRoPE(head_dim=8, embed_dim=32)
    k = torch.randn(2, 32, 4, 8)
    mu = torch.full((2, 32), 3.0)
    sigma = torch.ones(2, 32)

    out_l2r = rope.apply_k(
        k, height=4, width=8, mu=mu, sigma=sigma, direction=StereoDirection.LEFT_TO_RIGHT
    )
    out_r2l = rope.apply_k(
        k, height=4, width=8, mu=mu, sigma=sigma, direction=StereoDirection.RIGHT_TO_LEFT
    )

    assert not torch.allclose(out_l2r, out_r2l)
    assert torch.allclose(out_l2r[..., : rope.h_dim], out_r2l[..., : rope.h_dim], atol=1e-5)


def test_rectified_disparity_rope_q_shape_and_finite() -> None:
    rope = RectifiedDisparityRoPE(head_dim=8)
    q = torch.randn(2, 32, 4, 8)

    out = rope.apply_q(q, height=4, width=8)

    assert out.shape == q.shape
    assert out.isfinite().all()


def test_rectified_disparity_rope_k_with_disparity() -> None:
    rope = RectifiedDisparityRoPE(head_dim=8)
    k = torch.randn(2, 32, 4, 8)
    disp_zero = torch.zeros(2, 32)
    disp_nonzero = torch.full((2, 32), 5.0)
    sigma = torch.ones(2, 32)

    out_zero = rope.apply_k(
        k,
        height=4,
        width=8,
        disparity=disp_zero,
        sigma=sigma,
        direction=StereoDirection.LEFT_TO_RIGHT,
    )
    out_shifted = rope.apply_k(
        k,
        height=4,
        width=8,
        disparity=disp_nonzero,
        sigma=sigma,
        direction=StereoDirection.LEFT_TO_RIGHT,
    )

    assert out_zero.isfinite().all()
    assert out_shifted.isfinite().all()
    assert not torch.allclose(out_zero, out_shifted)


def test_rectified_disparity_rope_direction_flips() -> None:
    rope = RectifiedDisparityRoPE(head_dim=8)
    k = torch.randn(2, 32, 4, 8)
    disp = torch.full((2, 32), 3.0)
    sigma = torch.ones(2, 32)

    out_l2r = rope.apply_k(
        k,
        height=4,
        width=8,
        disparity=disp,
        sigma=sigma,
        direction=StereoDirection.LEFT_TO_RIGHT,
    )
    out_r2l = rope.apply_k(
        k,
        height=4,
        width=8,
        disparity=disp,
        sigma=sigma,
        direction=StereoDirection.RIGHT_TO_LEFT,
    )

    assert not torch.allclose(out_l2r, out_r2l)
    assert torch.allclose(out_l2r[..., : rope.h_dim], out_r2l[..., : rope.h_dim], atol=1e-5)


def test_rectified_disparity_rope_zero_disp_zero_sigma_matches_q() -> None:
    rope = RectifiedDisparityRoPE(head_dim=8)
    tensor = torch.randn(2, 32, 4, 8)
    disp = torch.zeros(2, 32)
    sigma = torch.zeros(2, 32)

    q_out = rope.apply_q(tensor, height=4, width=8)
    k_out = rope.apply_k(
        tensor,
        height=4,
        width=8,
        disparity=disp,
        sigma=sigma,
        direction=StereoDirection.LEFT_TO_RIGHT,
    )

    assert torch.allclose(q_out, k_out, atol=1e-5)


def test_rectified_disparity_rope_beta_scales_effect() -> None:
    rope_small = RectifiedDisparityRoPE(head_dim=8, init_beta=0.01)
    rope_large = RectifiedDisparityRoPE(head_dim=8, init_beta=1.0)

    q = torch.randn(2, 32, 4, 8)
    identity = q.clone()

    out_small = rope_small.apply_q(q, height=4, width=8)
    out_large = rope_large.apply_q(q, height=4, width=8)

    diff_small = (out_small - identity).abs().mean()
    diff_large = (out_large - identity).abs().mean()
    assert diff_small < diff_large


def test_rectified_disparity_rope_grid_cache_respects_width() -> None:
    rope = RectifiedDisparityRoPE(head_dim=8)
    q = torch.randn(2, 32, 4, 8)

    rope.apply_q(q, height=4, width=8)
    cols_8 = rope._cached_cols.clone()

    q_alt = torch.randn(2, 32, 4, 8)
    rope.apply_q(q_alt, height=8, width=4)
    cols_4 = rope._cached_cols.clone()

    assert not torch.equal(cols_8, cols_4)


def test_windowed_cross_attention_full_width_shape_and_finite() -> None:
    module = WindowedCrossAttention(embed_dim=32, num_heads=4, window_size=(4, -1))
    left = torch.randn(1, 32, 8, 16)
    right = torch.randn(1, 32, 8, 16)

    left_out, right_out = module(left, right)

    assert left_out.shape == left.shape
    assert right_out.shape == right.shape
    assert left_out.isfinite().all()
    assert right_out.isfinite().all()


def test_windowed_cross_attention_full_width_with_shift() -> None:
    module = WindowedCrossAttention(embed_dim=32, num_heads=4, window_size=(4, -1))
    left = torch.randn(1, 32, 8, 16)
    right = torch.randn(1, 32, 8, 16)

    left_out, right_out = module(left, right, shift=True)

    assert left_out.shape == left.shape
    assert right_out.shape == right.shape
    assert left_out.isfinite().all()
    assert right_out.isfinite().all()


def test_windowed_cross_attention_full_width_train_eval_match() -> None:
    torch.manual_seed(0)
    module = WindowedCrossAttention(embed_dim=32, num_heads=4, window_size=(4, -1))
    left = torch.randn(1, 32, 8, 16)
    right = torch.randn(1, 32, 8, 16)

    module.eval()
    with torch.no_grad():
        eval_left, eval_right = module(left, right)

    module.train()
    with torch.no_grad():
        train_left, train_right = module(left, right)

    assert torch.allclose(eval_left, train_left, atol=1e-4, rtol=1e-4)
    assert torch.allclose(eval_right, train_right, atol=1e-4, rtol=1e-4)


def test_windowed_cross_attention_gradients_flow_to_all_params() -> None:
    module = WindowedCrossAttention(embed_dim=32, num_heads=4, window_size=(4, 8))
    left = torch.randn(1, 32, 8, 16)
    right = torch.randn(1, 32, 8, 16)

    left_out, right_out = module(left, right, shift=True)
    torch.autograd.backward(left_out.sum() + right_out.sum())

    for name, param in module.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"no gradient for {name}"
